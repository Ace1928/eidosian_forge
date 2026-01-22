from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2CryptoReplaceFfxFpeConfig(_messages.Message):
    """Replaces an identifier with a surrogate using Format Preserving
  Encryption (FPE) with the FFX mode of operation; however when used in the
  `ReidentifyContent` API method, it serves the opposite function by reversing
  the surrogate back into the original identifier. The identifier must be
  encoded as ASCII. For a given crypto key and context, the same identifier
  will be replaced with the same surrogate. Identifiers must be at least two
  characters long. In the case that the identifier is the empty string, it
  will be skipped. See https://cloud.google.com/sensitive-data-
  protection/docs/pseudonymization to learn more. Note: We recommend using
  CryptoDeterministicConfig for all use cases which do not require preserving
  the input alphabet space and size, plus warrant referential integrity.

  Enums:
    CommonAlphabetValueValuesEnum: Common alphabets.

  Fields:
    commonAlphabet: Common alphabets.
    context: The 'tweak', a context may be used for higher security since the
      same identifier in two different contexts won't be given the same
      surrogate. If the context is not set, a default tweak will be used. If
      the context is set but: 1. there is no record present when transforming
      a given value or 1. the field is not present when transforming a given
      value, a default tweak will be used. Note that case (1) is expected when
      an `InfoTypeTransformation` is applied to both structured and
      unstructured `ContentItem`s. Currently, the referenced field may be of
      value type integer or string. The tweak is constructed as a sequence of
      bytes in big endian byte order such that: - a 64 bit integer is encoded
      followed by a single byte of value 1 - a string is encoded in UTF-8
      format followed by a single byte of value 2
    cryptoKey: Required. The key used by the encryption algorithm.
    customAlphabet: This is supported by mapping these to the alphanumeric
      characters that the FFX mode natively supports. This happens
      before/after encryption/decryption. Each character listed must appear
      only once. Number of characters must be in the range [2, 95]. This must
      be encoded as ASCII. The order of characters does not matter. The full
      list of allowed characters is:
      0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
      ~`!@#$%^&*()_-+={[}]|\\:;"'<,>.?/
    radix: The native way to select the alphabet. Must be in the range [2,
      95].
    surrogateInfoType: The custom infoType to annotate the surrogate with.
      This annotation will be applied to the surrogate by prefixing it with
      the name of the custom infoType followed by the number of characters
      comprising the surrogate. The following scheme defines the format:
      info_type_name(surrogate_character_count):surrogate For example, if the
      name of custom infoType is 'MY_TOKEN_INFO_TYPE' and the surrogate is
      'abc', the full replacement value will be: 'MY_TOKEN_INFO_TYPE(3):abc'
      This annotation identifies the surrogate when inspecting content using
      the custom infoType
      [`SurrogateType`](https://cloud.google.com/sensitive-data-
      protection/docs/reference/rest/v2/InspectConfig#surrogatetype). This
      facilitates reversal of the surrogate when it occurs in free text. In
      order for inspection to work properly, the name of this infoType must
      not occur naturally anywhere in your data; otherwise, inspection may
      find a surrogate that does not correspond to an actual identifier.
      Therefore, choose your custom infoType name carefully after considering
      what your data looks like. One way to select a name that has a high
      chance of yielding reliable detection is to include one or more unicode
      characters that are highly improbable to exist in your data. For
      example, assuming your data is entered from a regular ASCII keyboard,
      the symbol with the hex code point 29DD might be used like so:
      \\u29ddMY_TOKEN_TYPE
  """

    class CommonAlphabetValueValuesEnum(_messages.Enum):
        """Common alphabets.

    Values:
      FFX_COMMON_NATIVE_ALPHABET_UNSPECIFIED: Unused.
      NUMERIC: `[0-9]` (radix of 10)
      HEXADECIMAL: `[0-9A-F]` (radix of 16)
      UPPER_CASE_ALPHA_NUMERIC: `[0-9A-Z]` (radix of 36)
      ALPHA_NUMERIC: `[0-9A-Za-z]` (radix of 62)
    """
        FFX_COMMON_NATIVE_ALPHABET_UNSPECIFIED = 0
        NUMERIC = 1
        HEXADECIMAL = 2
        UPPER_CASE_ALPHA_NUMERIC = 3
        ALPHA_NUMERIC = 4
    commonAlphabet = _messages.EnumField('CommonAlphabetValueValuesEnum', 1)
    context = _messages.MessageField('GooglePrivacyDlpV2FieldId', 2)
    cryptoKey = _messages.MessageField('GooglePrivacyDlpV2CryptoKey', 3)
    customAlphabet = _messages.StringField(4)
    radix = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    surrogateInfoType = _messages.MessageField('GooglePrivacyDlpV2InfoType', 6)