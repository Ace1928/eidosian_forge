import requests
from wandb_gql import gql
from wandb.apis.attrs import Attrs
Create a service account for the team.

        Arguments:
            description: (str) A description for this service account

        Returns:
            The service account `Member` object, or None on failure
        