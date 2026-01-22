from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class NegativeCachingPolicyValue(_messages.Message):
    """Optional. A cache TTL for the specified HTTP status code.
    negative_caching must be enabled to configure `negative_caching_policy`.
    The following limitations apply: - Omitting the policy and leaving
    `negative_caching` enabled uses the default TTLs for each status code,
    defined in `negative_caching`. - TTLs must be >= `0` (where `0` is "always
    revalidate") and <= `86400s` (1 day) You can set only the following status
    codes: - HTTP redirection (`300`, `301`, `302`, `307`, or `308`) - Client
    error (`400`, `403`, `404`, `405`, `410`, `421`, or `451`) - Server error
    (`500`, `501`, `502`, `503`, or `504`) When you specify an explicit
    `negative_caching_policy`, ensure that you also specify a cache TTL for
    all response codes that you wish to cache. The CDNPolicy doesn't apply any
    default negative caching when a policy exists.

    Messages:
      AdditionalProperty: An additional property for a
        NegativeCachingPolicyValue object.

    Fields:
      additionalProperties: Additional properties of type
        NegativeCachingPolicyValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a NegativeCachingPolicyValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)