from pprint import pformat
from six import iteritems
import re
@admission_review_versions.setter
def admission_review_versions(self, admission_review_versions):
    """
        Sets the admission_review_versions of this V1beta1Webhook.
        AdmissionReviewVersions is an ordered list of preferred
        `AdmissionReview` versions the Webhook expects. API server will try to
        use first version in the list which it supports. If none of the versions
        specified in this list supported by API server, validation will fail for
        this object. If a persisted webhook configuration specifies allowed
        versions and does not include any versions known to the API Server,
        calls to the webhook will fail and be subject to the failure policy.
        Default to `['v1beta1']`.

        :param admission_review_versions: The admission_review_versions of this
        V1beta1Webhook.
        :type: list[str]
        """
    self._admission_review_versions = admission_review_versions