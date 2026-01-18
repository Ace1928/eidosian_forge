from direct.showbase.ShowBase import ShowBase
from panda3d.core import PointLight, AmbientLight, Vec4, Vec3
from panda3d.core import CollisionTraverser, CollisionNode
from panda3d.core import CollisionHandlerPusher, CollisionSphere
from panda3d.bullet import BulletWorld, BulletBoxShape, BulletRigidBodyNode
from panda3d.bullet import BulletSphereShape
import random
import logging

# Configure logging to the most detailed level possible
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class GameEnvironmentInitialization(ShowBase):
    def __init__(self):
        super().__init__()
        logging.debug(
            "GameEnvironmentInitialization: Superclass initialization complete."
        )

        self.configure_window_camera_and_background()
        self.initialize_lighting()
        self.configure_physics_world()
        self.initialize_collision_handling()
        self.construct_environmental_elements()
        self.schedule_regular_updates()
        self.set_frame_rate()

    def configure_window_camera_and_background(self):
        self.set_background_color(0.1, 0.1, 0.1, 1)
        self.cam.set_pos(0, -50, 20)
        self.cam.look_at(0, 0, 0)
        logging.info(
            "GameEnvironmentInitialization: Window, camera, and background color configured."
        )

    def initialize_lighting(self):
        self.configure_ambient_light()
        self.configure_point_light()

    def configure_ambient_light(self):
        ambient_light = AmbientLight("ambient_light")
        ambient_light.set_color(Vec4(0.2, 0.2, 0.2, 1))
        ambient_light_node = self.render.attach_new_node(ambient_light)
        self.render.set_light(ambient_light_node)
        logging.info("GameEnvironmentInitialization: Ambient light configured.")

    def configure_point_light(self):
        point_light = PointLight("point_light")
        point_light.set_color(Vec4(0.9, 0.9, 0.9, 1))
        point_light_node = self.render.attach_new_node(point_light)
        point_light_node.set_pos(10, -20, 20)
        self.render.set_light(point_light_node)
        logging.info("GameEnvironmentInitialization: Point light configured.")

    def configure_physics_world(self):
        self.world = BulletWorld()
        self.world.set_gravity(Vec3(0, 0, -9.81))
        logging.info(
            "GameEnvironmentInitialization: Physics world configured with gravity."
        )

    def initialize_collision_handling(self):
        self.traverser = CollisionTraverser()
        self.pusher = CollisionHandlerPusher()
        logging.info("GameEnvironmentInitialization: Collision handling initialized.")

    def construct_environmental_elements(self):
        self.create_ground()
        self.create_player()
        self.create_obstacles()

    def create_ground(self):
        shape = BulletBoxShape(Vec3(10, 10, 1))
        body = BulletRigidBodyNode("Ground")
        body.add_shape(shape)
        np = self.render.attach_new_node(body)
        np.set_pos(0, 0, -2)
        self.world.attach_rigid_body(body)
        logging.debug(
            "GameEnvironmentInitialization: Ground element created and positioned."
        )

    def create_player(self):
        shape = BulletSphereShape(1)
        body = BulletRigidBodyNode("Player")
        body.set_mass(1.0)
        body.add_shape(shape)
        self.player_np = self.render.attach_new_node(body)
        self.player_np.set_pos(0, 0, 2)
        self.world.attach_rigid_body(body)
        self.define_player_collision_sphere()

    def define_player_collision_sphere(self):
        coll_node = CollisionNode("player")
        coll_node.add_solid(CollisionSphere(0, 0, 0, 1))
        coll_np = self.player_np.attach_new_node(coll_node)
        self.traverser.add_collider(coll_np, self.pusher)
        logging.debug("GameEnvironmentInitialization: Player collision sphere defined.")

    def create_obstacles(self):
        for _ in range(10):
            x, y, z = random.uniform(-8, 8), random.uniform(-8, 8), 0
            shape = BulletBoxShape(Vec3(1, 1, 1))
            body = BulletRigidBodyNode("Obstacle")
            body.add_shape(shape)
            np = self.render.attach_new_node(body)
            np.set_pos(x, y, z)
            self.world.attach_rigid_body(body)
        logging.debug(
            "GameEnvironmentInitialization: Obstacles created and positioned."
        )

    def schedule_regular_updates(self):
        self.task_mgr.add(self.update_physics_and_logging, "update")
        logging.info("GameEnvironmentInitialization: Regular updates scheduled.")

    def set_frame_rate(self):
        globalClock.set_frame_rate(60)
        logging.info("GameEnvironmentInitialization: Frame rate set to 60 FPS.")

    def update_physics_and_logging(self, task):
        dt = globalClock.get_dt()
        self.world.do_physics(dt)
        player_pos = self.player_np.get_pos()
        logging.debug(
            f"GameEnvironmentInitialization: Physics updated for dt: {dt}, Player position: {player_pos}"
        )
        return task.cont


game_environment = GameEnvironmentInitialization()
game_environment.run()
logging.info(
    "GameEnvironmentInitialization: Game execution started and main loop running."
)
