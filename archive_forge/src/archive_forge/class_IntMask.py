import json
import netaddr
import re
class IntMask(Decoder):
    """Base class for Integer Mask decoder classes.

    It supports decoding a value/mask pair. The class has to be derived,
    and the size attribute must be set.
    """
    size = None

    def __init__(self, string):
        if not self.size:
            raise NotImplementedError('IntMask should be derived and size should be fixed')
        parts = string.split('/')
        if len(parts) > 1:
            self._value = int(parts[0], 0)
            self._mask = int(parts[1], 0)
            if self._mask.bit_length() > self.size:
                raise ValueError('Integer mask {} is bigger than size {}'.format(self._mask, self.size))
        else:
            self._value = int(parts[0], 0)
            self._mask = self.max_mask()
        if self._value.bit_length() > self.size:
            raise ValueError('Integer value {} is bigger than size {}'.format(self._value, self.size))

    @property
    def value(self):
        return self._value

    @property
    def mask(self):
        return self._mask

    def max_mask(self):
        return 2 ** self.size - 1

    def fully(self):
        """Returns True if it's fully masked."""
        return self._mask == self.max_mask()

    def __str__(self):
        if self.fully():
            return str(self._value)
        else:
            return '{}/{}'.format(hex(self._value), hex(self._mask))

    def __repr__(self):
        return "%s('%s')" % (self.__class__.__name__, self)

    def __eq__(self, other):
        """Equality operator.

        Both value and mask must be the same for the comparison to result True.
        This can be used to implement filters that expect a specific mask,
        e.g: ct.state = 0x1/0xff.

        Args:
            other (IntMask): Another IntMask to compare against.

        Returns:
            True if the other IntMask is the same as this one.
        """
        if isinstance(other, IntMask):
            return self.value == other.value and self.mask == other.mask
        elif isinstance(other, int):
            return self.value == other and self.mask == self.max_mask()
        else:
            raise ValueError('Cannot compare against ', other)

    def __contains__(self, other):
        """Contains operator.

        Args:
            other (int or IntMask): Another integer or fully-masked IntMask
            to compare against.

        Returns:
            True if the other integer or fully-masked IntMask is
            contained in this IntMask.

        Example:
            0x1 in IntMask("0xf1/0xff"): True
            0x1 in IntMask("0xf1/0x0f"): True
            0x1 in IntMask("0xf1/0xf0"): False
        """
        if isinstance(other, IntMask):
            if other.fully():
                return other.value in self
            else:
                raise ValueError('Comparing non fully-masked IntMasks is not supported')
        else:
            return other & self._mask == self._value & self._mask

    def dict(self):
        return {'value': self._value, 'mask': self._mask}

    def to_json(self):
        return self.dict()