from __future__ import (absolute_import, division, print_function)
import sys as _sys
from collections.abc import Sequence
from ansible.module_utils.six import text_type
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
class AnsibleVaultEncryptedUnicode(Sequence, AnsibleBaseYAMLObject):
    """Unicode like object that is not evaluated (decrypted) until it needs to be"""
    __UNSAFE__ = True
    __ENCRYPTED__ = True
    yaml_tag = u'!vault'

    @classmethod
    def from_plaintext(cls, seq, vault, secret):
        if not vault:
            raise vault.AnsibleVaultError('Error creating AnsibleVaultEncryptedUnicode, invalid vault (%s) provided' % vault)
        ciphertext = vault.encrypt(seq, secret)
        avu = cls(ciphertext)
        avu.vault = vault
        return avu

    def __init__(self, ciphertext):
        """A AnsibleUnicode with a Vault attribute that can decrypt it.

        ciphertext is a byte string (str on PY2, bytestring on PY3).

        The .data attribute is a property that returns the decrypted plaintext
        of the ciphertext as a PY2 unicode or PY3 string object.
        """
        super(AnsibleVaultEncryptedUnicode, self).__init__()
        self.vault = None
        self._ciphertext = to_bytes(ciphertext)

    @property
    def data(self):
        if not self.vault:
            return to_text(self._ciphertext)
        return to_text(self.vault.decrypt(self._ciphertext, obj=self))

    @data.setter
    def data(self, value):
        self._ciphertext = to_bytes(value)

    def is_encrypted(self):
        return self.vault and self.vault.is_encrypted(self._ciphertext)

    def __eq__(self, other):
        if self.vault:
            return other == self.data
        return False

    def __ne__(self, other):
        if self.vault:
            return other != self.data
        return True

    def __reversed__(self):
        return to_text(self[::-1], errors='surrogate_or_strict')

    def __str__(self):
        return to_native(self.data, errors='surrogate_or_strict')

    def __unicode__(self):
        return to_text(self.data, errors='surrogate_or_strict')

    def encode(self, encoding=None, errors=None):
        return to_bytes(self.data, encoding=encoding, errors=errors)

    def __repr__(self):
        return repr(self.data)

    def __int__(self, base=10):
        return int(self.data, base=base)

    def __float__(self):
        return float(self.data)

    def __complex__(self):
        return complex(self.data)

    def __hash__(self):
        return hash(self.data)

    def __lt__(self, string):
        if isinstance(string, AnsibleVaultEncryptedUnicode):
            return self.data < string.data
        return self.data < string

    def __le__(self, string):
        if isinstance(string, AnsibleVaultEncryptedUnicode):
            return self.data <= string.data
        return self.data <= string

    def __gt__(self, string):
        if isinstance(string, AnsibleVaultEncryptedUnicode):
            return self.data > string.data
        return self.data > string

    def __ge__(self, string):
        if isinstance(string, AnsibleVaultEncryptedUnicode):
            return self.data >= string.data
        return self.data >= string

    def __contains__(self, char):
        if isinstance(char, AnsibleVaultEncryptedUnicode):
            char = char.data
        return char in self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __getslice__(self, start, end):
        start = max(start, 0)
        end = max(end, 0)
        return self.data[start:end]

    def __add__(self, other):
        if isinstance(other, AnsibleVaultEncryptedUnicode):
            return self.data + other.data
        elif isinstance(other, text_type):
            return self.data + other
        return self.data + to_text(other)

    def __radd__(self, other):
        if isinstance(other, text_type):
            return other + self.data
        return to_text(other) + self.data

    def __mul__(self, n):
        return self.data * n
    __rmul__ = __mul__

    def __mod__(self, args):
        return self.data % args

    def __rmod__(self, template):
        return to_text(template) % self

    def capitalize(self):
        return self.data.capitalize()

    def casefold(self):
        return self.data.casefold()

    def center(self, width, *args):
        return self.data.center(width, *args)

    def count(self, sub, start=0, end=_sys.maxsize):
        if isinstance(sub, AnsibleVaultEncryptedUnicode):
            sub = sub.data
        return self.data.count(sub, start, end)

    def endswith(self, suffix, start=0, end=_sys.maxsize):
        return self.data.endswith(suffix, start, end)

    def expandtabs(self, tabsize=8):
        return self.data.expandtabs(tabsize)

    def find(self, sub, start=0, end=_sys.maxsize):
        if isinstance(sub, AnsibleVaultEncryptedUnicode):
            sub = sub.data
        return self.data.find(sub, start, end)

    def format(self, *args, **kwds):
        return self.data.format(*args, **kwds)

    def format_map(self, mapping):
        return self.data.format_map(mapping)

    def index(self, sub, start=0, end=_sys.maxsize):
        return self.data.index(sub, start, end)

    def isalpha(self):
        return self.data.isalpha()

    def isalnum(self):
        return self.data.isalnum()

    def isascii(self):
        return self.data.isascii()

    def isdecimal(self):
        return self.data.isdecimal()

    def isdigit(self):
        return self.data.isdigit()

    def isidentifier(self):
        return self.data.isidentifier()

    def islower(self):
        return self.data.islower()

    def isnumeric(self):
        return self.data.isnumeric()

    def isprintable(self):
        return self.data.isprintable()

    def isspace(self):
        return self.data.isspace()

    def istitle(self):
        return self.data.istitle()

    def isupper(self):
        return self.data.isupper()

    def join(self, seq):
        return self.data.join(seq)

    def ljust(self, width, *args):
        return self.data.ljust(width, *args)

    def lower(self):
        return self.data.lower()

    def lstrip(self, chars=None):
        return self.data.lstrip(chars)
    maketrans = str.maketrans

    def partition(self, sep):
        return self.data.partition(sep)

    def replace(self, old, new, maxsplit=-1):
        if isinstance(old, AnsibleVaultEncryptedUnicode):
            old = old.data
        if isinstance(new, AnsibleVaultEncryptedUnicode):
            new = new.data
        return self.data.replace(old, new, maxsplit)

    def rfind(self, sub, start=0, end=_sys.maxsize):
        if isinstance(sub, AnsibleVaultEncryptedUnicode):
            sub = sub.data
        return self.data.rfind(sub, start, end)

    def rindex(self, sub, start=0, end=_sys.maxsize):
        return self.data.rindex(sub, start, end)

    def rjust(self, width, *args):
        return self.data.rjust(width, *args)

    def rpartition(self, sep):
        return self.data.rpartition(sep)

    def rstrip(self, chars=None):
        return self.data.rstrip(chars)

    def split(self, sep=None, maxsplit=-1):
        return self.data.split(sep, maxsplit)

    def rsplit(self, sep=None, maxsplit=-1):
        return self.data.rsplit(sep, maxsplit)

    def splitlines(self, keepends=False):
        return self.data.splitlines(keepends)

    def startswith(self, prefix, start=0, end=_sys.maxsize):
        return self.data.startswith(prefix, start, end)

    def strip(self, chars=None):
        return self.data.strip(chars)

    def swapcase(self):
        return self.data.swapcase()

    def title(self):
        return self.data.title()

    def translate(self, *args):
        return self.data.translate(*args)

    def upper(self):
        return self.data.upper()

    def zfill(self, width):
        return self.data.zfill(width)