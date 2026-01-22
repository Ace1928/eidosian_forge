from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildBazelRemoteExecutionV2Action(_messages.Message):
    """An `Action` captures all the information about an execution which is
  required to reproduce it. `Action`s are the core component of the
  [Execution] service. A single `Action` represents a repeatable action that
  can be performed by the execution service. `Action`s can be succinctly
  identified by the digest of their wire format encoding and, once an `Action`
  has been executed, will be cached in the action cache. Future requests can
  then use the cached result rather than needing to run afresh. When a server
  completes execution of an Action, it MAY choose to cache the result in the
  ActionCache unless `do_not_cache` is `true`. Clients SHOULD expect the
  server to do so. By default, future calls to Execute the same `Action` will
  also serve their results from the cache. Clients must take care to
  understand the caching behaviour. Ideally, all `Action`s will be
  reproducible so that serving a result from cache is always desirable and
  correct.

  Fields:
    commandDigest: The digest of the Command to run, which MUST be present in
      the ContentAddressableStorage.
    doNotCache: If true, then the `Action`'s result cannot be cached, and in-
      flight requests for the same `Action` may not be merged.
    inputRootDigest: The digest of the root Directory for the input files. The
      files in the directory tree are available in the correct location on the
      build machine before the command is executed. The root directory, as
      well as every subdirectory and content blob referred to, MUST be in the
      ContentAddressableStorage.
    platform: The optional platform requirements for the execution
      environment. The server MAY choose to execute the action on any worker
      satisfying the requirements, so the client SHOULD ensure that running
      the action on any such worker will have the same result. A detailed
      lexicon for this can be found in the accompanying platform.md. New in
      version 2.2: clients SHOULD set these platform properties as well as
      those in the Command. Servers SHOULD prefer those set here.
    salt: An optional additional salt value used to place this `Action` into a
      separate cache namespace from other instances having the same field
      contents. This salt typically comes from operational configuration
      specific to sources such as repo and service configuration, and allows
      disowning an entire set of ActionResults that might have been poisoned
      by buggy software or tool failures.
    timeout: A timeout after which the execution should be killed. If the
      timeout is absent, then the client is specifying that the execution
      should continue as long as the server will let it. The server SHOULD
      impose a timeout if the client does not specify one, however, if the
      client does specify a timeout that is longer than the server's maximum
      timeout, the server MUST reject the request. The timeout is only
      intended to cover the "execution" of the specified action and not time
      in queue nor any overheads before or after execution such as marshalling
      inputs/outputs. The server SHOULD avoid including time spent the client
      doesn't have control over, and MAY extend or reduce the timeout to
      account for delays or speedups that occur during execution itself (e.g.,
      lazily loading data from the Content Addressable Storage, live migration
      of virtual machines, emulation overhead). The timeout is a part of the
      Action message, and therefore two `Actions` with different timeouts are
      different, even if they are otherwise identical. This is because, if
      they were not, running an `Action` with a lower timeout than is required
      might result in a cache hit from an execution run with a longer timeout,
      hiding the fact that the timeout is too short. By encoding it directly
      in the `Action`, a lower timeout will result in a cache miss and the
      execution timeout will fail immediately, rather than whenever the cache
      entry gets evicted.
  """
    commandDigest = _messages.MessageField('BuildBazelRemoteExecutionV2Digest', 1)
    doNotCache = _messages.BooleanField(2)
    inputRootDigest = _messages.MessageField('BuildBazelRemoteExecutionV2Digest', 3)
    platform = _messages.MessageField('BuildBazelRemoteExecutionV2Platform', 4)
    salt = _messages.BytesField(5)
    timeout = _messages.StringField(6)