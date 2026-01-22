import json
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from wandb_gql import Client, gql
import wandb
from wandb.apis import public
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.paginator import Paginator
from wandb.errors.term import termlog
class ArtifactType:

    def __init__(self, client: Client, entity: str, project: str, type_name: str, attrs: Optional[Mapping[str, Any]]=None):
        self.client = client
        self.entity = entity
        self.project = project
        self.type = type_name
        self._attrs = attrs
        if self._attrs is None:
            self.load()

    def load(self):
        query = gql('\n        query ProjectArtifactType(\n            $entityName: String!,\n            $projectName: String!,\n            $artifactTypeName: String!\n        ) {\n            project(name: $projectName, entityName: $entityName) {\n                artifactType(name: $artifactTypeName) {\n                    id\n                    name\n                    description\n                    createdAt\n                }\n            }\n        }\n        ')
        response: Optional[Mapping[str, Any]] = self.client.execute(query, variable_values={'entityName': self.entity, 'projectName': self.project, 'artifactTypeName': self.type})
        if response is None or response.get('project') is None or response['project'].get('artifactType') is None:
            raise ValueError('Could not find artifact type %s' % self.type)
        self._attrs = response['project']['artifactType']
        return self._attrs

    @property
    def id(self):
        return self._attrs['id']

    @property
    def name(self):
        return self._attrs['name']

    @normalize_exceptions
    def collections(self, per_page=50):
        """Artifact collections."""
        return ArtifactCollections(self.client, self.entity, self.project, self.type)

    def collection(self, name):
        return ArtifactCollection(self.client, self.entity, self.project, name, self.type)

    def __repr__(self):
        return f'<ArtifactType {self.type}>'