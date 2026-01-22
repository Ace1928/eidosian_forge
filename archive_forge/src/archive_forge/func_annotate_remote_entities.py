import abc
import json
from copy import deepcopy
from inspect import signature
from typing import Dict, List, Union
from dataclasses import dataclass
import ray
from ray.util import placement_group
from ray.util.annotations import DeveloperAPI
def annotate_remote_entities(self, entities: List[RemoteRayEntity]) -> List[Union[RemoteRayEntity]]:
    """Return remote ray entities (tasks/actors) to use the acquired resources.

        The first entity will be associated with the first bundle, the second
        entity will be associated with the second bundle, etc.

        Args:
            entities: Remote Ray entities to annotate with the acquired resources.
        """
    bundles = self.resource_request.bundles
    num_bundles = len(bundles) + int(self.resource_request.head_bundle_is_empty)
    if len(entities) > num_bundles:
        raise RuntimeError(f'The number of callables to annotate ({len(entities)}) cannot exceed the number of available bundles ({num_bundles}).')
    annotated = []
    if self.resource_request.head_bundle_is_empty:
        annotated.append(self._annotate_remote_entity(entities[0], {}, bundle_index=0))
        entities = entities[1:]
    for i, (entity, bundle) in enumerate(zip(entities, bundles)):
        annotated.append(self._annotate_remote_entity(entity, bundle, bundle_index=i))
    return annotated