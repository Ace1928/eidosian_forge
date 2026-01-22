from wandb_gql import gql
from wandb.apis import public
from wandb.apis.attrs import Attrs
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.paginator import Paginator
from wandb.sdk.lib import ipython
@normalize_exceptions
def artifacts_types(self, per_page=50):
    return public.ArtifactTypes(self.client, self.entity, self.name)