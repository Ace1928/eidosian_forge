from typing import Optional, Tuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
A wrapper which converts old-style envs to valid modern envs.

        Some information may be lost in the conversion, so we recommend updating your
        environment.

        Args:
            old_env: The old MultiAgentEnv to wrap. Implemented with the old API.
            render_mode: The render mode to use when rendering the environment,
                passed automatically to `env.render()`.
        