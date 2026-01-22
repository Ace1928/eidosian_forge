import ipaddress
from functools import lru_cache
class DjangoRangeDumper(RangeDumper):
    """A Range dumper customized for Django."""

    def upgrade(self, obj, format):
        dumper = super().upgrade(obj, format)
        if dumper is not self and dumper.oid == TSRANGE_OID:
            dumper.oid = TSTZRANGE_OID
        return dumper