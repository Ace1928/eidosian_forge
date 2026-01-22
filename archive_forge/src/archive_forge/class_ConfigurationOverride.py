from __future__ import annotations
import dataclasses
from typing import Optional
from torch.onnx._internal.diagnostics.infra.sarif import (
@dataclasses.dataclass
class ConfigurationOverride(object):
    """Information about how a specific rule or notification was reconfigured at runtime."""
    configuration: _reporting_configuration.ReportingConfiguration = dataclasses.field(metadata={'schema_property_name': 'configuration'})
    descriptor: _reporting_descriptor_reference.ReportingDescriptorReference = dataclasses.field(metadata={'schema_property_name': 'descriptor'})
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(default=None, metadata={'schema_property_name': 'properties'})