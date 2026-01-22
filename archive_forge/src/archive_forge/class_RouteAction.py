from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouteAction(_messages.Message):
    """The actions (such as rewrites, redirects, CORS header injection, and
  header modification) to take for a given route match.

  Fields:
    cdnPolicy: Optional. The policy to use for defining caching and signed
      request behavior for requests that match this route.
    corsPolicy: Optional. The Cross-Origin Resource Sharing (CORS) policy for
      requests that match this route.
    urlRewrite: Optional. The URL rewrite configuration for requests that
      match this route.
    wasmAction: Optional. A WasmAction resource in the format:
      `projects/{project}/locations/{location}/wasmActions/{wasm_action}`
  """
    cdnPolicy = _messages.MessageField('CDNPolicy', 1)
    corsPolicy = _messages.MessageField('CORSPolicy', 2)
    urlRewrite = _messages.MessageField('UrlRewrite', 3)
    wasmAction = _messages.StringField(4)