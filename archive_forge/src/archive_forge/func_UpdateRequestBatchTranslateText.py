from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def UpdateRequestBatchTranslateText(unused_instance_ref, args, request):
    """The hook to inject content into the batch translate request."""
    messages = apis.GetMessagesModule(SPEECH_API, _GetApiVersion(args))
    batch_translate_text_request = messages.BatchTranslateTextRequest()
    project = properties.VALUES.core.project.GetOrFail()
    request.parent = 'projects/{}/locations/{}'.format(project, args.zone)
    batch_translate_text_request.sourceLanguageCode = args.source_language
    batch_translate_text_request.targetLanguageCodes = args.target_language_codes
    batch_translate_text_request.outputConfig = messages.OutputConfig(gcsDestination=messages.GcsDestination(outputUriPrefix=args.destination))
    batch_translate_text_request.inputConfigs = [messages.InputConfig(gcsSource=messages.GcsSource(inputUri=k), mimeType=v if v else None) for k, v in sorted(args.source.items())]
    if args.IsSpecified('models'):
        batch_translate_text_request.models = messages.BatchTranslateTextRequest.ModelsValue(additionalProperties=[messages.BatchTranslateTextRequest.ModelsValue.AdditionalProperty(key=k, value='projects/{}/locations/{}/models/{}'.format(project, args.zone, v)) for k, v in sorted(args.models.items())])
    if args.IsSpecified('glossaries'):
        additional_properties = [messages.BatchTranslateTextRequest.GlossariesValue.AdditionalProperty(key=k, value=messages.TranslateTextGlossaryConfig(glossary='projects/{}/locations/{}/glossaries/{}'.format(project, args.zone, v))) for k, v in sorted(args.glossaries.items())]
        batch_translate_text_request.glossaries = messages.BatchTranslateTextRequest.GlossariesValue(additionalProperties=additional_properties)
    request.batchTranslateTextRequest = batch_translate_text_request
    return request