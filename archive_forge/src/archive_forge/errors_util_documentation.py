from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
Raises error for user input mismatched with command resource type.

  Example message:

  "gcloud storage buckets" create only accepts bucket URLs.
  Example: "gs://bucket"
  Received: "gs://user-bucket/user-object.txt"

  Args:
    command_list (list[str]): The command being run. Can be gotten from an
      argparse object with `args.command_path`.
    expected_resource_type (str): Raise an error because we did not get this.
    example: (str): An example of a URL to a resource with the correct type.
    url (StorageUrl): The erroneous URL received.

  Raises:
    InvalidUrlError: Explains that the user entered a URL for the wrong type
      of resource.
  