from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSaasacceleratorManagementProvidersV1PerSliSloEligibility(_messages.Message):
    """PerSliSloEligibility is a mapping from an SLI name to eligibility.

  Messages:
    EligibilitiesValue: An entry in the eligibilities map specifies an
      eligibility for a particular SLI for the given instance. The SLI key in
      the name must be a valid SLI name specified in the Eligibility Exporter
      binary flags otherwise an error will be emitted by Eligibility Exporter
      and the oncaller will be alerted. If an SLI has been defined in the
      binary flags but the eligibilities map does not contain it, the
      corresponding SLI time series will not be emitted by the Eligibility
      Exporter. This ensures a smooth rollout and compatibility between the
      data produced by different versions of the Eligibility Exporters. If
      eligibilities map contains a key for an SLI which has not been declared
      in the binary flags, there will be an error message emitted in the
      Eligibility Exporter log and the metric for the SLI in question will not
      be emitted.

  Fields:
    eligibilities: An entry in the eligibilities map specifies an eligibility
      for a particular SLI for the given instance. The SLI key in the name
      must be a valid SLI name specified in the Eligibility Exporter binary
      flags otherwise an error will be emitted by Eligibility Exporter and the
      oncaller will be alerted. If an SLI has been defined in the binary flags
      but the eligibilities map does not contain it, the corresponding SLI
      time series will not be emitted by the Eligibility Exporter. This
      ensures a smooth rollout and compatibility between the data produced by
      different versions of the Eligibility Exporters. If eligibilities map
      contains a key for an SLI which has not been declared in the binary
      flags, there will be an error message emitted in the Eligibility
      Exporter log and the metric for the SLI in question will not be emitted.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class EligibilitiesValue(_messages.Message):
        """An entry in the eligibilities map specifies an eligibility for a
    particular SLI for the given instance. The SLI key in the name must be a
    valid SLI name specified in the Eligibility Exporter binary flags
    otherwise an error will be emitted by Eligibility Exporter and the
    oncaller will be alerted. If an SLI has been defined in the binary flags
    but the eligibilities map does not contain it, the corresponding SLI time
    series will not be emitted by the Eligibility Exporter. This ensures a
    smooth rollout and compatibility between the data produced by different
    versions of the Eligibility Exporters. If eligibilities map contains a key
    for an SLI which has not been declared in the binary flags, there will be
    an error message emitted in the Eligibility Exporter log and the metric
    for the SLI in question will not be emitted.

    Messages:
      AdditionalProperty: An additional property for a EligibilitiesValue
        object.

    Fields:
      additionalProperties: Additional properties of type EligibilitiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a EligibilitiesValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudSaasacceleratorManagementProvidersV1SloEligibility
          attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudSaasacceleratorManagementProvidersV1SloEligibility', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    eligibilities = _messages.MessageField('EligibilitiesValue', 1)