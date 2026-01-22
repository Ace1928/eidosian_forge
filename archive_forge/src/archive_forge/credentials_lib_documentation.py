from apitools.base.py import GceAssertionCredentials
Shim for backwards-compatibility for moving GCE credentials.

oauth2client loads credentials classes based on the module name where
they were created; this means that moving GceAssertionCredentials from
here to third_party requires a shim mapping the old name to the new
one. Once loaded, the credential will be re-serialized with the new
path, meaning that we can (at some point) consider removing this file.
