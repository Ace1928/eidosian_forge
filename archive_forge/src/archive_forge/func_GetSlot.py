from . import encode
from . import number_types as N
def GetSlot(self, slot, d, validator_flags):
    N.enforce_number(slot, N.VOffsetTFlags)
    if validator_flags is not None:
        N.enforce_number(d, validator_flags)
    off = self.Offset(slot)
    if off == 0:
        return d
    return self.Get(validator_flags, self.Pos + off)