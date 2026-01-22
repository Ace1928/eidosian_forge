import functools
class _IPv6Constants:
    _linklocal_network = IPv6Network('fe80::/10')
    _multicast_network = IPv6Network('ff00::/8')
    _private_networks = [IPv6Network('::1/128'), IPv6Network('::/128'), IPv6Network('::ffff:0:0/96'), IPv6Network('100::/64'), IPv6Network('2001::/23'), IPv6Network('2001:2::/48'), IPv6Network('2001:db8::/32'), IPv6Network('2001:10::/28'), IPv6Network('fc00::/7'), IPv6Network('fe80::/10')]
    _reserved_networks = [IPv6Network('::/8'), IPv6Network('100::/8'), IPv6Network('200::/7'), IPv6Network('400::/6'), IPv6Network('800::/5'), IPv6Network('1000::/4'), IPv6Network('4000::/3'), IPv6Network('6000::/3'), IPv6Network('8000::/3'), IPv6Network('A000::/3'), IPv6Network('C000::/3'), IPv6Network('E000::/4'), IPv6Network('F000::/5'), IPv6Network('F800::/6'), IPv6Network('FE00::/9')]
    _sitelocal_network = IPv6Network('fec0::/10')