from __future__ import annotations
import dataclasses
from typing import Optional
from torch.onnx._internal.diagnostics.infra.sarif import (
@dataclasses.dataclass
class SpecialLocations(object):
    """Defines locations of special significance to SARIF consumers."""
    display_base: Optional[_artifact_location.ArtifactLocation] = dataclasses.field(default=None, metadata={'schema_property_name': 'displayBase'})
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(default=None, metadata={'schema_property_name': 'properties'})