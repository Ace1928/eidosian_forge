from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.utils import seeding
import ray
class CartPoleWithRemoteParamServer(CartPoleEnv):
    """CartPoleMassEnv varies the weights of the cart and the pole."""

    def __init__(self, env_config):
        self.env_config = env_config
        super().__init__()
        self._handler = ray.get_actor(env_config.get('param_server', 'param-server'))
        self.rng_seed = None
        self.np_random, _ = seeding.np_random(self.rng_seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng_seed = int(seed)
            self.np_random, _ = seeding.np_random(seed)
            print(f'Seeding env (worker={self.env_config.worker_index}) with {seed}')
        params = ray.get(self._handler.get_params.remote(self.np_random))
        new_seed = int(self.np_random.integers(0, 1000000) if not self.rng_seed else self.rng_seed)
        self.np_random, _ = seeding.np_random(new_seed)
        print(f'Env worker-idx={self.env_config.worker_index} mass={params['MASSCART']}')
        self.masscart = params['MASSCART']
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length
        return super().reset()