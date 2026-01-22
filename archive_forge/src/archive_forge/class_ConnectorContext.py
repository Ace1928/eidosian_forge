import abc
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union
import gymnasium as gym
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
class ConnectorContext:
    """Data bits that may be needed for running connectors.

    Note(jungong) : we need to be really careful with the data fields here.
    E.g., everything needs to be serializable, in case we need to fetch them
    in a remote setting.
    """

    def __init__(self, config: AlgorithmConfigDict=None, initial_states: List[TensorType]=None, observation_space: gym.Space=None, action_space: gym.Space=None, view_requirements: Dict[str, ViewRequirement]=None, is_policy_recurrent: bool=False):
        """Construct a ConnectorContext instance.

        Args:
            initial_states: States that are used for constructing
                the initial input dict for RNN models. [] if a model is not recurrent.
            action_space_struct: a policy's action space, in python
                data format. E.g., python dict instead of DictSpace, python tuple
                instead of TupleSpace.
        """
        self.config = config or {}
        self.initial_states = initial_states or []
        self.observation_space = observation_space
        self.action_space = action_space
        self.view_requirements = view_requirements
        self.is_policy_recurrent = is_policy_recurrent

    @staticmethod
    def from_policy(policy: 'Policy') -> 'ConnectorContext':
        """Build ConnectorContext from a given policy.

        Args:
            policy: Policy

        Returns:
            A ConnectorContext instance.
        """
        return ConnectorContext(config=policy.config, initial_states=policy.get_initial_state(), observation_space=policy.observation_space, action_space=policy.action_space, view_requirements=policy.view_requirements, is_policy_recurrent=policy.is_recurrent())