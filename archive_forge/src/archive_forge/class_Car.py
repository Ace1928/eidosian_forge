import math
import Box2D
import numpy as np
from gym.error import DependencyNotInstalled
class Car:

    def __init__(self, world, init_angle, init_x, init_y):
        self.world: Box2D.b2World = world
        self.hull: Box2D.b2Body = self.world.CreateDynamicBody(position=(init_x, init_y), angle=init_angle, fixtures=[fixtureDef(shape=polygonShape(vertices=[(x * SIZE, y * SIZE) for x, y in HULL_POLY1]), density=1.0), fixtureDef(shape=polygonShape(vertices=[(x * SIZE, y * SIZE) for x, y in HULL_POLY2]), density=1.0), fixtureDef(shape=polygonShape(vertices=[(x * SIZE, y * SIZE) for x, y in HULL_POLY3]), density=1.0), fixtureDef(shape=polygonShape(vertices=[(x * SIZE, y * SIZE) for x, y in HULL_POLY4]), density=1.0)])
        self.hull.color = (0.8, 0.0, 0.0)
        self.wheels = []
        self.fuel_spent = 0.0
        WHEEL_POLY = [(-WHEEL_W, +WHEEL_R), (+WHEEL_W, +WHEEL_R), (+WHEEL_W, -WHEEL_R), (-WHEEL_W, -WHEEL_R)]
        for wx, wy in WHEELPOS:
            front_k = 1.0 if wy > 0 else 1.0
            w = self.world.CreateDynamicBody(position=(init_x + wx * SIZE, init_y + wy * SIZE), angle=init_angle, fixtures=fixtureDef(shape=polygonShape(vertices=[(x * front_k * SIZE, y * front_k * SIZE) for x, y in WHEEL_POLY]), density=0.1, categoryBits=32, maskBits=1, restitution=0.0))
            w.wheel_rad = front_k * WHEEL_R * SIZE
            w.color = WHEEL_COLOR
            w.gas = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0
            w.omega = 0.0
            w.skid_start = None
            w.skid_particle = None
            rjd = revoluteJointDef(bodyA=self.hull, bodyB=w, localAnchorA=(wx * SIZE, wy * SIZE), localAnchorB=(0, 0), enableMotor=True, enableLimit=True, maxMotorTorque=180 * 900 * SIZE * SIZE, motorSpeed=0, lowerAngle=-0.4, upperAngle=+0.4)
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            w.userData = w
            self.wheels.append(w)
        self.drawlist = self.wheels + [self.hull]
        self.particles = []

    def gas(self, gas):
        """control: rear wheel drive

        Args:
            gas (float): How much gas gets applied. Gets clipped between 0 and 1.
        """
        gas = np.clip(gas, 0, 1)
        for w in self.wheels[2:4]:
            diff = gas - w.gas
            if diff > 0.1:
                diff = 0.1
            w.gas += diff

    def brake(self, b):
        """control: brake

        Args:
            b (0..1): Degree to which the brakes are applied. More than 0.9 blocks the wheels to zero rotation"""
        for w in self.wheels:
            w.brake = b

    def steer(self, s):
        """control: steer

        Args:
            s (-1..1): target position, it takes time to rotate steering wheel from side-to-side"""
        self.wheels[0].steer = s
        self.wheels[1].steer = s

    def step(self, dt):
        for w in self.wheels:
            dir = np.sign(w.steer - w.joint.angle)
            val = abs(w.steer - w.joint.angle)
            w.joint.motorSpeed = dir * min(50.0 * val, 3.0)
            grass = True
            friction_limit = FRICTION_LIMIT * 0.6
            for tile in w.tiles:
                friction_limit = max(friction_limit, FRICTION_LIMIT * tile.road_friction)
                grass = False
            forw = w.GetWorldVector((0, 1))
            side = w.GetWorldVector((1, 0))
            v = w.linearVelocity
            vf = forw[0] * v[0] + forw[1] * v[1]
            vs = side[0] * v[0] + side[1] * v[1]
            w.omega += dt * ENGINE_POWER * w.gas / WHEEL_MOMENT_OF_INERTIA / (abs(w.omega) + 5.0)
            self.fuel_spent += dt * ENGINE_POWER * w.gas
            if w.brake >= 0.9:
                w.omega = 0
            elif w.brake > 0:
                BRAKE_FORCE = 15
                dir = -np.sign(w.omega)
                val = BRAKE_FORCE * w.brake
                if abs(val) > abs(w.omega):
                    val = abs(w.omega)
                w.omega += dir * val
            w.phase += w.omega * dt
            vr = w.omega * w.wheel_rad
            f_force = -vf + vr
            p_force = -vs
            f_force *= 205000 * SIZE * SIZE
            p_force *= 205000 * SIZE * SIZE
            force = np.sqrt(np.square(f_force) + np.square(p_force))
            if abs(force) > 2.0 * friction_limit:
                if w.skid_particle and w.skid_particle.grass == grass and (len(w.skid_particle.poly) < 30):
                    w.skid_particle.poly.append((w.position[0], w.position[1]))
                elif w.skid_start is None:
                    w.skid_start = w.position
                else:
                    w.skid_particle = self._create_particle(w.skid_start, w.position, grass)
                    w.skid_start = None
            else:
                w.skid_start = None
                w.skid_particle = None
            if abs(force) > friction_limit:
                f_force /= force
                p_force /= force
                force = friction_limit
                f_force *= force
                p_force *= force
            w.omega -= dt * f_force * w.wheel_rad / WHEEL_MOMENT_OF_INERTIA
            w.ApplyForceToCenter((p_force * side[0] + f_force * forw[0], p_force * side[1] + f_force * forw[1]), True)

    def draw(self, surface, zoom, translation, angle, draw_particles=True):
        import pygame.draw
        if draw_particles:
            for p in self.particles:
                poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in p.poly]
                poly = [(coords[0] * zoom + translation[0], coords[1] * zoom + translation[1]) for coords in poly]
                pygame.draw.lines(surface, color=p.color, points=poly, width=2, closed=False)
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                path = [(coords[0], coords[1]) for coords in path]
                path = [pygame.math.Vector2(c).rotate_rad(angle) for c in path]
                path = [(coords[0] * zoom + translation[0], coords[1] * zoom + translation[1]) for coords in path]
                color = [int(c * 255) for c in obj.color]
                pygame.draw.polygon(surface, color=color, points=path)
                if 'phase' not in obj.__dict__:
                    continue
                a1 = obj.phase
                a2 = obj.phase + 1.2
                s1 = math.sin(a1)
                s2 = math.sin(a2)
                c1 = math.cos(a1)
                c2 = math.cos(a2)
                if s1 > 0 and s2 > 0:
                    continue
                if s1 > 0:
                    c1 = np.sign(c1)
                if s2 > 0:
                    c2 = np.sign(c2)
                white_poly = [(-WHEEL_W * SIZE, +WHEEL_R * c1 * SIZE), (+WHEEL_W * SIZE, +WHEEL_R * c1 * SIZE), (+WHEEL_W * SIZE, +WHEEL_R * c2 * SIZE), (-WHEEL_W * SIZE, +WHEEL_R * c2 * SIZE)]
                white_poly = [trans * v for v in white_poly]
                white_poly = [(coords[0], coords[1]) for coords in white_poly]
                white_poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in white_poly]
                white_poly = [(coords[0] * zoom + translation[0], coords[1] * zoom + translation[1]) for coords in white_poly]
                pygame.draw.polygon(surface, color=WHEEL_WHITE, points=white_poly)

    def _create_particle(self, point1, point2, grass):

        class Particle:
            pass
        p = Particle()
        p.color = WHEEL_COLOR if not grass else MUD_COLOR
        p.ttl = 1
        p.poly = [(point1[0], point1[1]), (point2[0], point2[1])]
        p.grass = grass
        self.particles.append(p)
        while len(self.particles) > 30:
            self.particles.pop(0)
        return p

    def destroy(self):
        self.world.DestroyBody(self.hull)
        self.hull = None
        for w in self.wheels:
            self.world.DestroyBody(w)
        self.wheels = []