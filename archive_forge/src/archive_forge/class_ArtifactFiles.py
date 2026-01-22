import json
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from wandb_gql import Client, gql
import wandb
from wandb.apis import public
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.paginator import Paginator
from wandb.errors.term import termlog
class ArtifactFiles(Paginator):
    QUERY = gql('\n        query ArtifactFiles(\n            $entityName: String!,\n            $projectName: String!,\n            $artifactTypeName: String!,\n            $artifactName: String!\n            $fileNames: [String!],\n            $fileCursor: String,\n            $fileLimit: Int = 50\n        ) {\n            project(name: $projectName, entityName: $entityName) {\n                artifactType(name: $artifactTypeName) {\n                    artifact(name: $artifactName) {\n                        ...ArtifactFilesFragment\n                    }\n                }\n            }\n        }\n        %s\n    ' % ARTIFACT_FILES_FRAGMENT)

    def __init__(self, client: Client, artifact: 'wandb.Artifact', names: Optional[Sequence[str]]=None, per_page: int=50):
        self.artifact = artifact
        variables = {'entityName': artifact.source_entity, 'projectName': artifact.source_project, 'artifactTypeName': artifact.type, 'artifactName': artifact.source_name, 'fileNames': names}
        if not client.version_supported('0.12.21'):
            self.QUERY = gql(self.QUERY.loc.source.body.replace('storagePath\n', ''))
        super().__init__(client, variables, per_page)

    @property
    def path(self):
        return [self.artifact.entity, self.artifact.project, self.artifact.name]

    @property
    def length(self):
        return self.artifact.file_count

    @property
    def more(self):
        if self.last_response:
            return self.last_response['project']['artifactType']['artifact']['files']['pageInfo']['hasNextPage']
        else:
            return True

    @property
    def cursor(self):
        if self.last_response:
            return self.last_response['project']['artifactType']['artifact']['files']['edges'][-1]['cursor']
        else:
            return None

    def update_variables(self):
        self.variables.update({'fileLimit': self.per_page, 'fileCursor': self.cursor})

    def convert_objects(self):
        return [public.File(self.client, r['node']) for r in self.last_response['project']['artifactType']['artifact']['files']['edges']]

    def __repr__(self):
        return '<ArtifactFiles {} ({})>'.format('/'.join(self.path), len(self))