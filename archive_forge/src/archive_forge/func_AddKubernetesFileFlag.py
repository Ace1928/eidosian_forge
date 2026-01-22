from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddKubernetesFileFlag():
    return base.Argument('--from-k8s-manifest', help='The path to a Kubernetes manifest, which Cloud Deploy will use to generate a skaffold.yaml file for you (for example, foo/bar/k8.yaml). The generated Skaffold file will be available in the Google Cloud Storage source staging directory (see --gcs-source-staging-dir flag) after the release is complete.')