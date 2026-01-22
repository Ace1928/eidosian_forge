import argparse
from ...utils.dataclasses import (
from ..menu import BulletMenu
def _convert_dynamo_backend(value):
    value = int(value)
    return DynamoBackend(DYNAMO_BACKENDS[value]).value