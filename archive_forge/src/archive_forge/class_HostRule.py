from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HostRule(_messages.Message):
    """The hostname configured for the EdgeCacheService. A `HostRule` value
  associates a hostname (or hostnames) with a set of routing rules, which
  define configuration based on the path and header.

  Fields:
    description: Optional. A human-readable description of the `HostRule`
      value.
    hosts: Required. A list of host patterns to match. Host patterns must be
      valid hostnames. Ports are not allowed. Wildcard hosts are supported in
      the suffix or prefix form. `*` matches any string of `([a-z0-9-.]*)`. It
      does not match an empty string. When multiple hosts are specified, hosts
      are matched in the following priority: 1. Exact domain names:
      `www.foo.com`. 2. Suffix domain wildcards: `*.foo.com` or
      `*-bar.foo.com`. 3. Prefix domain wildcards: `foo.*` or `foo-*`. 4.
      Special wildcard `*` matching any domain. The wildcard doesn't match the
      empty string. For example, `*-bar.foo.com` matches `baz-bar.foo.com` but
      not `-bar.foo.com`. The longest wildcards match first. Only a single
      host in the entire service can match on ``*``. A domain must be unique
      across all configured hosts within a service. Hosts are matched against
      the HTTP `Host` header, or for HTTP/2 and HTTP/3, the `:authority`
      header, in the incoming request. You can specify up to 10 hosts.
    pathMatcher: Required. The name of the PathMatcher associated with this
      `HostRule`.
  """
    description = _messages.StringField(1)
    hosts = _messages.StringField(2, repeated=True)
    pathMatcher = _messages.StringField(3)