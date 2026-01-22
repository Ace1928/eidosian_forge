from googlecloudsdk.api_lib.audit_manager import util
def Gcs_uri_to_eligible_destination(self, gcs_uri):
    dest = self.messages.EligibleDestination()
    dest.eligibleGcsBucket = gcs_uri
    return dest