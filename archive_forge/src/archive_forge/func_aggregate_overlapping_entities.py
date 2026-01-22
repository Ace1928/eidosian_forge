import types
import warnings
from typing import List, Optional, Tuple, Union
import numpy as np
from ..models.bert.tokenization_bert import BasicTokenizer
from ..utils import (
from .base import ArgumentHandler, ChunkPipeline, Dataset, build_pipeline_init_args
def aggregate_overlapping_entities(self, entities):
    if len(entities) == 0:
        return entities
    entities = sorted(entities, key=lambda x: x['start'])
    aggregated_entities = []
    previous_entity = entities[0]
    for entity in entities:
        if previous_entity['start'] <= entity['start'] < previous_entity['end']:
            current_length = entity['end'] - entity['start']
            previous_length = previous_entity['end'] - previous_entity['start']
            if current_length > previous_length:
                previous_entity = entity
            elif current_length == previous_length and entity['score'] > previous_entity['score']:
                previous_entity = entity
        else:
            aggregated_entities.append(previous_entity)
            previous_entity = entity
    aggregated_entities.append(previous_entity)
    return aggregated_entities