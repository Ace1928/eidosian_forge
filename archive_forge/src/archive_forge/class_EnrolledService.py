from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EnrolledService(_messages.Message):
    """Represents the enrollment of a cloud resource into a specific service.

  Enums:
    EnrollmentLevelValueValuesEnum: The enrollment level of the service.

  Fields:
    cloudProduct: The product for which Access Approval will be enrolled.
      Allowed values are listed below (case-sensitive): * all * GA * App
      Engine * Artifact Registry * BigQuery * Certificate Authority Service *
      Cloud Bigtable * Cloud Key Management Service * Compute Engine * Cloud
      Composer * Cloud Dataflow * Cloud Dataproc * Cloud DLP * Cloud EKM *
      Cloud Firestore * Cloud HSM * Cloud Identity and Access Management *
      Cloud Logging * Cloud NAT * Cloud Pub/Sub * Cloud Spanner * Cloud SQL *
      Cloud Storage * Eventarc * Google Kubernetes Engine * Organization
      Policy Serivice * Persistent Disk * Resource Manager * Secret Manager *
      Speaker ID Note: These values are supported as input for legacy
      purposes, but will not be returned from the API. * all * ga-only *
      appengine.googleapis.com * artifactregistry.googleapis.com *
      bigquery.googleapis.com * bigtable.googleapis.com *
      container.googleapis.com * cloudkms.googleapis.com *
      cloudresourcemanager.googleapis.com * cloudsql.googleapis.com *
      compute.googleapis.com * dataflow.googleapis.com *
      dataproc.googleapis.com * dlp.googleapis.com * iam.googleapis.com *
      logging.googleapis.com * orgpolicy.googleapis.com *
      pubsub.googleapis.com * spanner.googleapis.com *
      secretmanager.googleapis.com * speakerid.googleapis.com *
      storage.googleapis.com Calls to UpdateAccessApprovalSettings using 'all'
      or any of the XXX.googleapis.com will be translated to the associated
      product name ('all', 'App Engine', etc.). Note: 'all' will enroll the
      resource in all products supported at both 'GA' and 'Preview' levels.
      More information about levels of support is available at
      https://cloud.google.com/access-approval/docs/supported-services
    enrollmentLevel: The enrollment level of the service.
  """

    class EnrollmentLevelValueValuesEnum(_messages.Enum):
        """The enrollment level of the service.

    Values:
      ENROLLMENT_LEVEL_UNSPECIFIED: Default value for proto, shouldn't be
        used.
      BLOCK_ALL: Service is enrolled in Access Approval for all requests
    """
        ENROLLMENT_LEVEL_UNSPECIFIED = 0
        BLOCK_ALL = 1
    cloudProduct = _messages.StringField(1)
    enrollmentLevel = _messages.EnumField('EnrollmentLevelValueValuesEnum', 2)