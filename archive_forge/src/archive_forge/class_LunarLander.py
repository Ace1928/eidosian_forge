import math
import warnings
from typing import TYPE_CHECKING, Optional
import numpy as np
import gym
from gym import error, spaces
from gym.error import DependencyNotInstalled
from gym.utils import EzPickle, colorize
from gym.utils.step_api_compatibility import step_api_compatibility
class LunarLander(gym.Env, EzPickle):
    """
    ### Description
    This environment is a classic rocket trajectory optimization problem.
    According to Pontryagin's maximum principle, it is optimal to fire the
    engine at full throttle or turn it off. This is the reason why this
    environment has discrete actions: engine on or off.

    There are two environment versions: discrete or continuous.
    The landing pad is always at coordinates (0,0). The coordinates are the
    first two numbers in the state vector.
    Landing outside of the landing pad is possible. Fuel is infinite, so an agent
    can learn to fly and then land on its first attempt.

    To see a heuristic landing, run:
    ```
    python gym/envs/box2d/lunar_lander.py
    ```
    <!-- To play yourself, run: -->
    <!-- python examples/agents/keyboard_agent.py LunarLander-v2 -->

    ### Action Space
    There are four discrete actions available: do nothing, fire left
    orientation engine, fire main engine, fire right orientation engine.

    ### Observation Space
    The state is an 8-dimensional vector: the coordinates of the lander in `x` & `y`, its linear
    velocities in `x` & `y`, its angle, its angular velocity, and two booleans
    that represent whether each leg is in contact with the ground or not.

    ### Rewards
    After every step a reward is granted. The total reward of an episode is the
    sum of the rewards for all the steps within that episode.

    For each step, the reward:
    - is increased/decreased the closer/further the lander is to the landing pad.
    - is increased/decreased the slower/faster the lander is moving.
    - is decreased the more the lander is tilted (angle not horizontal).
    - is increased by 10 points for each leg that is in contact with the ground.
    - is decreased by 0.03 points each frame a side engine is firing.
    - is decreased by 0.3 points each frame the main engine is firing.

    The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.

    An episode is considered a solution if it scores at least 200 points.

    ### Starting State
    The lander starts at the top center of the viewport with a random initial
    force applied to its center of mass.

    ### Episode Termination
    The episode finishes if:
    1) the lander crashes (the lander body gets in contact with the moon);
    2) the lander gets outside of the viewport (`x` coordinate is greater than 1);
    3) the lander is not awake. From the [Box2D docs](https://box2d.org/documentation/md__d_1__git_hub_box2d_docs_dynamics.html#autotoc_md61),
        a body which is not awake is a body which doesn't move and doesn't
        collide with any other body:
    > When Box2D determines that a body (or group of bodies) has come to rest,
    > the body enters a sleep state which has very little CPU overhead. If a
    > body is awake and collides with a sleeping body, then the sleeping body
    > wakes up. Bodies will also wake up if a joint or contact attached to
    > them is destroyed.

    ### Arguments
    To use to the _continuous_ environment, you need to specify the
    `continuous=True` argument like below:
    ```python
    import gym
    env = gym.make(
        "LunarLander-v2",
        continuous: bool = False,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
    )
    ```
    If `continuous=True` is passed, continuous actions (corresponding to the throttle of the engines) will be used and the
    action space will be `Box(-1, +1, (2,), dtype=np.float32)`.
    The first coordinate of an action determines the throttle of the main engine, while the second
    coordinate specifies the throttle of the lateral boosters.
    Given an action `np.array([main, lateral])`, the main engine will be turned off completely if
    `main < 0` and the throttle scales affinely from 50% to 100% for `0 <= main <= 1` (in particular, the
    main engine doesn't work  with less than 50% power).
    Similarly, if `-0.5 < lateral < 0.5`, the lateral boosters will not fire at all. If `lateral < -0.5`, the left
    booster will fire, and if `lateral > 0.5`, the right booster will fire. Again, the throttle scales affinely
    from 50% to 100% between -1 and -0.5 (and 0.5 and 1, respectively).

    `gravity` dictates the gravitational constant, this is bounded to be within 0 and -12.

    If `enable_wind=True` is passed, there will be wind effects applied to the lander.
    The wind is generated using the function `tanh(sin(2 k (t+C)) + sin(pi k (t+C)))`.
    `k` is set to 0.01.
    `C` is sampled randomly between -9999 and 9999.

    `wind_power` dictates the maximum magnitude of linear wind applied to the craft. The recommended value for `wind_power` is between 0.0 and 20.0.
    `turbulence_power` dictates the maximum magnitude of rotational wind applied to the craft. The recommended value for `turbulence_power` is between 0.0 and 2.0.

    ### Version History
    - v2: Count energy spent and in v0.24, added turbulance with wind power and turbulence_power parameters
    - v1: Legs contact with ground added in state vector; contact with ground
        give +10 reward points, and -10 if then lose contact; reward
        renormalized to 200; harder initial random push.
    - v0: Initial version

    <!-- ### References -->

    ### Credits
    Created by Oleg Klimov
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}

    def __init__(self, render_mode: Optional[str]=None, continuous: bool=False, gravity: float=-10.0, enable_wind: bool=False, wind_power: float=15.0, turbulence_power: float=1.5):
        EzPickle.__init__(self, render_mode, continuous, gravity, enable_wind, wind_power, turbulence_power)
        assert -12.0 < gravity and gravity < 0.0, f'gravity (current value: {gravity}) must be between -12 and 0'
        self.gravity = gravity
        if 0.0 > wind_power or wind_power > 20.0:
            warnings.warn(colorize(f'WARN: wind_power value is recommended to be between 0.0 and 20.0, (current value: {wind_power})', 'yellow'))
        self.wind_power = wind_power
        if 0.0 > turbulence_power or turbulence_power > 2.0:
            warnings.warn(colorize(f'WARN: turbulence_power value is recommended to be between 0.0 and 2.0, (current value: {turbulence_power})', 'yellow'))
        self.turbulence_power = turbulence_power
        self.enable_wind = enable_wind
        self.wind_idx = np.random.randint(-9999, 9999)
        self.torque_idx = np.random.randint(-9999, 9999)
        self.screen: pygame.Surface = None
        self.clock = None
        self.isopen = True
        self.world = Box2D.b2World(gravity=(0, gravity))
        self.moon = None
        self.lander: Optional[Box2D.b2Body] = None
        self.particles = []
        self.prev_reward = None
        self.continuous = continuous
        low = np.array([-1.5, -1.5, -5.0, -5.0, -math.pi, -5.0, -0.0, -0.0]).astype(np.float32)
        high = np.array([1.5, 1.5, 5.0, 5.0, math.pi, 5.0, 1.0, 1.0]).astype(np.float32)
        self.observation_space = spaces.Box(low, high)
        if self.continuous:
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(4)
        self.render_mode = render_mode

    def _destroy(self):
        if not self.moon:
            return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None
        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE
        CHUNKS = 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [0.33 * (height[i - 1] + height[i + 0] + height[i + 1]) for i in range(CHUNKS)]
        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)
        initial_y = VIEWPORT_H / SCALE
        self.lander: Box2D.b2Body = self.world.CreateDynamicBody(position=(VIEWPORT_W / SCALE / 2, initial_y), angle=0.0, fixtures=fixtureDef(shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]), density=5.0, friction=0.1, categoryBits=16, maskBits=1, restitution=0.0))
        self.lander.color1 = (128, 102, 230)
        self.lander.color2 = (77, 77, 128)
        self.lander.ApplyForceToCenter((self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)), True)
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(position=(VIEWPORT_W / SCALE / 2 - i * LEG_AWAY / SCALE, initial_y), angle=i * 0.05, fixtures=fixtureDef(shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)), density=1.0, restitution=0.0, categoryBits=32, maskBits=1))
            leg.ground_contact = False
            leg.color1 = (128, 102, 230)
            leg.color2 = (77, 77, 128)
            rjd = revoluteJointDef(bodyA=self.lander, bodyB=leg, localAnchorA=(0, 0), localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE), enableMotor=True, enableLimit=True, maxMotorTorque=LEG_SPRING_TORQUE, motorSpeed=+0.3 * i)
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)
        self.drawlist = [self.lander] + self.legs
        if self.render_mode == 'human':
            self.render()
        return (self.step(np.array([0, 0]) if self.continuous else 0)[0], {})

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(position=(x, y), angle=0.0, fixtures=fixtureDef(shape=circleShape(radius=2 / SCALE, pos=(0, 0)), density=mass, friction=0.1, categoryBits=256, maskBits=1, restitution=0.3))
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def step(self, action):
        assert self.lander is not None
        assert self.lander is not None, 'You forgot to call reset()'
        if self.enable_wind and (not (self.legs[0].ground_contact or self.legs[1].ground_contact)):
            wind_mag = math.tanh(math.sin(0.02 * self.wind_idx) + math.sin(math.pi * 0.01 * self.wind_idx)) * self.wind_power
            self.wind_idx += 1
            self.lander.ApplyForceToCenter((wind_mag, 0.0), True)
            torque_mag = math.tanh(math.sin(0.02 * self.torque_idx) + math.sin(math.pi * 0.01 * self.torque_idx)) * self.turbulence_power
            self.torque_idx += 1
            self.lander.ApplyTorque(torque_mag, True)
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(action), f'{action!r} ({type(action)}) invalid '
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]
        m_power = 0.0
        if self.continuous and action[0] > 0.0 or (not self.continuous and action == 2):
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
            oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)
            p.ApplyLinearImpulse((ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse((-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power), impulse_pos, True)
        s_power = 0.0
        if self.continuous and np.abs(action[1]) > 0.5 or (not self.continuous and action in [1, 3]):
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                direction = action - 2
                s_power = 1.0
            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            impulse_pos = (self.lander.position[0] + ox - tip[0] * 17 / SCALE, self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE)
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse((ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse((-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power), impulse_pos, True)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [(pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2), (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2), vel.x * (VIEWPORT_W / SCALE / 2) / FPS, vel.y * (VIEWPORT_H / SCALE / 2) / FPS, self.lander.angle, 20.0 * self.lander.angularVelocity / FPS, 1.0 if self.legs[0].ground_contact else 0.0, 1.0 if self.legs[1].ground_contact else 0.0]
        assert len(state) == 8
        reward = 0
        shaping = -100 * np.sqrt(state[0] * state[0] + state[1] * state[1]) - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3]) - 100 * abs(state[4]) + 10 * state[6] + 10 * state[7]
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping
        reward -= m_power * 0.3
        reward -= s_power * 0.03
        terminated = False
        if self.game_over or abs(state[0]) >= 1.0:
            terminated = True
            reward = -100
        if not self.lander.awake:
            terminated = True
            reward = +100
        if self.render_mode == 'human':
            self.render()
        return (np.array(state, dtype=np.float32), reward, terminated, False, {})

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(f'You are calling render method without specifying any render mode. You can specify the render_mode at initialization, e.g. gym("{self.spec.id}", render_mode="rgb_array")')
            return
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled('pygame is not installed, run `pip install gym[box2d]`')
        if self.screen is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        pygame.transform.scale(self.surf, (SCALE, SCALE))
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())
        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (int(max(0.2, 0.15 + obj.ttl) * 255), int(max(0.2, 0.5 * obj.ttl) * 255), int(max(0.2, 0.5 * obj.ttl) * 255))
            obj.color2 = (int(max(0.2, 0.15 + obj.ttl) * 255), int(max(0.2, 0.5 * obj.ttl) * 255), int(max(0.2, 0.5 * obj.ttl) * 255))
        self._clean_particles(False)
        for p in self.sky_polys:
            scaled_poly = []
            for coord in p:
                scaled_poly.append((coord[0] * SCALE, coord[1] * SCALE))
            pygame.draw.polygon(self.surf, (0, 0, 0), scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, (0, 0, 0))
        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(self.surf, color=obj.color1, center=trans * f.shape.pos * SCALE, radius=f.shape.radius * SCALE)
                    pygame.draw.circle(self.surf, color=obj.color2, center=trans * f.shape.pos * SCALE, radius=f.shape.radius * SCALE)
                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                    gfxdraw.aapolygon(self.surf, path, obj.color1)
                    pygame.draw.aalines(self.surf, color=obj.color2, points=path, closed=True)
                for x in [self.helipad_x1, self.helipad_x2]:
                    x = x * SCALE
                    flagy1 = self.helipad_y * SCALE
                    flagy2 = flagy1 + 50
                    pygame.draw.line(self.surf, color=(255, 255, 255), start_pos=(x, flagy1), end_pos=(x, flagy2), width=1)
                    pygame.draw.polygon(self.surf, color=(204, 204, 0), points=[(x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5)])
                    gfxdraw.aapolygon(self.surf, [(x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5)], (204, 204, 0))
        self.surf = pygame.transform.flip(self.surf, False, True)
        if self.render_mode == 'human':
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata['render_fps'])
            pygame.display.flip()
        elif self.render_mode == 'rgb_array':
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False