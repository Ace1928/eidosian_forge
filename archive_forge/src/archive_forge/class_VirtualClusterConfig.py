from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VirtualClusterConfig(_messages.Message):
    """The Dataproc cluster config for a cluster that does not directly control
  the underlying compute resources, such as a Dataproc-on-GKE cluster
  (https://cloud.google.com/dataproc/docs/guides/dpgke/dataproc-gke-overview).

  Fields:
    auxiliaryServicesConfig: Optional. Configuration of auxiliary services
      used by this cluster.
    kubernetesClusterConfig: Required. The configuration for running the
      Dataproc cluster on Kubernetes.
    stagingBucket: Optional. A Cloud Storage bucket used to stage job
      dependencies, config files, and job driver console output. If you do not
      specify a staging bucket, Cloud Dataproc will determine a Cloud Storage
      location (US, ASIA, or EU) for your cluster's staging bucket according
      to the Compute Engine zone where your cluster is deployed, and then
      create and manage this project-level, per-location bucket (see Dataproc
      staging and temp buckets
      (https://cloud.google.com/dataproc/docs/concepts/configuring-
      clusters/staging-bucket)). This field requires a Cloud Storage bucket
      name, not a gs://... URI to a Cloud Storage bucket.
  """
    auxiliaryServicesConfig = _messages.MessageField('AuxiliaryServicesConfig', 1)
    kubernetesClusterConfig = _messages.MessageField('KubernetesClusterConfig', 2)
    stagingBucket = _messages.StringField(3)