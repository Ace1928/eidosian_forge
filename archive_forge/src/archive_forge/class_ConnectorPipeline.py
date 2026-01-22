import abc
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union
import gymnasium as gym
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
class ConnectorPipeline(abc.ABC):
    """Utility class for quick manipulation of a connector pipeline."""

    def __init__(self, ctx: ConnectorContext, connectors: List[Connector]):
        self.connectors = connectors

    def in_training(self):
        for c in self.connectors:
            c.in_training()

    def in_eval(self):
        for c in self.connectors:
            c.in_eval()

    def remove(self, name: str):
        """Remove a connector by <name>

        Args:
            name: name of the connector to be removed.
        """
        idx = -1
        for i, c in enumerate(self.connectors):
            if c.__class__.__name__ == name:
                idx = i
                break
        if idx >= 0:
            del self.connectors[idx]
            logger.info(f'Removed connector {name} from {self.__class__.__name__}.')
        else:
            logger.warning(f'Trying to remove a non-existent connector {name}.')

    def insert_before(self, name: str, connector: Connector):
        """Insert a new connector before connector <name>

        Args:
            name: name of the connector before which a new connector
                will get inserted.
            connector: a new connector to be inserted.
        """
        idx = -1
        for idx, c in enumerate(self.connectors):
            if c.__class__.__name__ == name:
                break
        if idx < 0:
            raise ValueError(f'Can not find connector {name}')
        self.connectors.insert(idx, connector)
        logger.info(f'Inserted {connector.__class__.__name__} before {name} to {self.__class__.__name__}.')

    def insert_after(self, name: str, connector: Connector):
        """Insert a new connector after connector <name>

        Args:
            name: name of the connector after which a new connector
                will get inserted.
            connector: a new connector to be inserted.
        """
        idx = -1
        for idx, c in enumerate(self.connectors):
            if c.__class__.__name__ == name:
                break
        if idx < 0:
            raise ValueError(f'Can not find connector {name}')
        self.connectors.insert(idx + 1, connector)
        logger.info(f'Inserted {connector.__class__.__name__} after {name} to {self.__class__.__name__}.')

    def prepend(self, connector: Connector):
        """Append a new connector at the beginning of a connector pipeline.

        Args:
            connector: a new connector to be appended.
        """
        self.connectors.insert(0, connector)
        logger.info(f'Added {connector.__class__.__name__} to the beginning of {self.__class__.__name__}.')

    def append(self, connector: Connector):
        """Append a new connector at the end of a connector pipeline.

        Args:
            connector: a new connector to be appended.
        """
        self.connectors.append(connector)
        logger.info(f'Added {connector.__class__.__name__} to the end of {self.__class__.__name__}.')

    def __str__(self, indentation: int=0):
        return '\n'.join([' ' * indentation + self.__class__.__name__] + [c.__str__(indentation + 4) for c in self.connectors])

    def __getitem__(self, key: Union[str, int, type]):
        """Returns a list of connectors that fit 'key'.

        If key is a number n, we return a list with the nth element of this pipeline.
        If key is a Connector class or a string matching the class name of a
        Connector class, we return a list of all connectors in this pipeline matching
        the specified class.

        Args:
            key: The key to index by

        Returns: The Connector at index `key`.
        """
        if not isinstance(key, str):
            if isinstance(key, slice):
                raise NotImplementedError('Slicing of ConnectorPipeline is currently not supported.')
            elif isinstance(key, int):
                return [self.connectors[key]]
            elif isinstance(key, type):
                results = []
                for c in self.connectors:
                    if issubclass(c.__class__, key):
                        results.append(c)
                return results
            else:
                raise NotImplementedError('Indexing by {} is currently not supported.'.format(type(key)))
        results = []
        for c in self.connectors:
            if c.__class__.__name__ == key:
                results.append(c)
        return results