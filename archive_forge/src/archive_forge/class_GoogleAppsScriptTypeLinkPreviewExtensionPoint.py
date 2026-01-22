from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsScriptTypeLinkPreviewExtensionPoint(_messages.Message):
    """The configuration for a trigger that fires when a user types or pastes a
  link from a third-party or non-Google service into a Google Docs, Sheets, or
  Slides file.

  Messages:
    LocalizedLabelTextValue: Optional. A map of `labelText` to localize into
      other languages. Format the language in [ISO
      639](https://wikipedia.org/wiki/ISO_639_macrolanguage) and the
      country/region in [ISO 3166](https://wikipedia.org/wiki/ISO_3166),
      separated by a hyphen `-`. For example, `en-US`. If a user's locale is
      present in the map's keys, the user sees the localized version of the
      `labelText`.

  Fields:
    labelText: Required. The text for an example smart chip that prompts users
      to preview the link, such as `Example: Support case`. This text is
      static and displays before users execute the add-on.
    localizedLabelText: Optional. A map of `labelText` to localize into other
      languages. Format the language in [ISO
      639](https://wikipedia.org/wiki/ISO_639_macrolanguage) and the
      country/region in [ISO 3166](https://wikipedia.org/wiki/ISO_3166),
      separated by a hyphen `-`. For example, `en-US`. If a user's locale is
      present in the map's keys, the user sees the localized version of the
      `labelText`.
    logoUrl: Optional. The icon that displays in the smart chip and preview
      card. If omitted, the add-on uses its toolbar icon,
      [`logoUrl`](https://developers.google.com/workspace/add-ons/reference/re
      st/v1/projects.deployments#CommonAddOnManifest.FIELDS.logoUrl).
    patterns: Required. An array of URL patterns that trigger the add-on to
      preview links.
    runFunction: Required. Endpoint to execute when a link preview is
      triggered.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LocalizedLabelTextValue(_messages.Message):
        """Optional. A map of `labelText` to localize into other languages.
    Format the language in [ISO
    639](https://wikipedia.org/wiki/ISO_639_macrolanguage) and the
    country/region in [ISO 3166](https://wikipedia.org/wiki/ISO_3166),
    separated by a hyphen `-`. For example, `en-US`. If a user's locale is
    present in the map's keys, the user sees the localized version of the
    `labelText`.

    Messages:
      AdditionalProperty: An additional property for a LocalizedLabelTextValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        LocalizedLabelTextValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LocalizedLabelTextValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    labelText = _messages.StringField(1)
    localizedLabelText = _messages.MessageField('LocalizedLabelTextValue', 2)
    logoUrl = _messages.StringField(3)
    patterns = _messages.MessageField('GoogleAppsScriptTypeUriPattern', 4, repeated=True)
    runFunction = _messages.StringField(5)