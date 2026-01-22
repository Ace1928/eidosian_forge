import netaddr
class AddressConverter(object):

    def __init__(self, addr, strat, fallback=None, **kwargs):
        self._addr = addr
        self._strat = strat
        self._fallback = fallback
        self._addr_kwargs = kwargs

    def text_to_bin(self, text):
        try:
            return self._addr(text, **self._addr_kwargs).packed
        except Exception as e:
            if self._fallback is None:
                raise e
            ip = self._fallback(text, **self._addr_kwargs)
            return (ip.ip.packed, ip.netmask.packed)

    def bin_to_text(self, bin):
        return str(self._addr(self._strat.packed_to_int(bin), **self._addr_kwargs))