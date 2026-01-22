from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class GPUDevice:
    """
    Describes a single graphics processor (GPU).
    """
    vendor_id: float
    device_id: float
    vendor_string: str
    device_string: str
    driver_vendor: str
    driver_version: str
    sub_sys_id: typing.Optional[float] = None
    revision: typing.Optional[float] = None

    def to_json(self):
        json = dict()
        json['vendorId'] = self.vendor_id
        json['deviceId'] = self.device_id
        json['vendorString'] = self.vendor_string
        json['deviceString'] = self.device_string
        json['driverVendor'] = self.driver_vendor
        json['driverVersion'] = self.driver_version
        if self.sub_sys_id is not None:
            json['subSysId'] = self.sub_sys_id
        if self.revision is not None:
            json['revision'] = self.revision
        return json

    @classmethod
    def from_json(cls, json):
        return cls(vendor_id=float(json['vendorId']), device_id=float(json['deviceId']), vendor_string=str(json['vendorString']), device_string=str(json['deviceString']), driver_vendor=str(json['driverVendor']), driver_version=str(json['driverVersion']), sub_sys_id=float(json['subSysId']) if 'subSysId' in json else None, revision=float(json['revision']) if 'revision' in json else None)