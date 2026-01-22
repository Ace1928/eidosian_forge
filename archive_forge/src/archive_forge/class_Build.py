from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA)
class Build(base.Command):
    """Builds a flex template file from the specified parameters."""
    detailed_help = {'DESCRIPTION': 'Builds a flex template file from the specified parameters.', 'EXAMPLES': '          To build and store a flex template JSON file, run:\n\n            $ {command} gs://template-file-gcs-path --image=gcr://image-path               --metadata-file=/local/path/to/metadata.json --sdk-language=JAVA\n\n          If using prebuilt template image from private registry, run:\n\n            $ {command} gs://template-file-gcs-path               --image=private.registry.com:3000/image-path               --image-repository-username-secret-id="projects/test-project/secrets/username-secret"\n              --image-repository-password-secret-id="projects/test-project/secrets/password-secret/versions/latest"\n              --metadata-file=metadata.json\n              --sdk-language=JAVA\n\n          To build the template image and flex template JSON file, run:\n\n            $ {command} gs://template-file-gcs-path               --image-gcr-path=gcr://path-to-store-image               --jar=path/to/pipeline.jar --jar=path/to/dependency.jar               --env=FLEX_TEMPLATE_JAVA_MAIN_CLASS=classpath               --flex-template-base-image=JAVA11               --metadata-file=/local/path/to/metadata.json --sdk-language=JAVA\n          '}

    @staticmethod
    def Args(parser):
        _CommonArgs(parser)

    def Run(self, args):
        return _CommonRun(args)