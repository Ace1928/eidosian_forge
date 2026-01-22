import ast
import json
import sys
import urllib
from wandb_gql import gql
import wandb
from wandb.apis import public
from wandb.apis.attrs import Attrs
from wandb.apis.paginator import Paginator
from wandb.sdk.lib import ipython
class BetaReport(Attrs):
    """BetaReport is a class associated with reports created in wandb.

    WARNING: this API will likely change in a future release

    Attributes:
        name (string): report name
        description (string): report description;
        user (User): the user that created the report
        spec (dict): the spec off the report;
        updated_at (string): timestamp of last update
    """

    def __init__(self, client, attrs, entity=None, project=None):
        self.client = client
        self.project = project
        self.entity = entity
        self.query_generator = public.QueryGenerator()
        super().__init__(dict(attrs))
        self._attrs['spec'] = json.loads(self._attrs['spec'])

    @property
    def sections(self):
        return self.spec['panelGroups']

    def runs(self, section, per_page=50, only_selected=True):
        run_set_idx = section.get('openRunSet', 0)
        run_set = section['runSets'][run_set_idx]
        order = self.query_generator.key_to_server_path(run_set['sort']['key'])
        if run_set['sort'].get('ascending'):
            order = '+' + order
        else:
            order = '-' + order
        filters = self.query_generator.filter_to_mongo(run_set['filters'])
        if only_selected:
            filters['$or'][0]['$and'].append({'name': {'$in': run_set['selections']['tree']}})
        return public.Runs(self.client, self.entity, self.project, filters=filters, order=order, per_page=per_page)

    @property
    def updated_at(self):
        return self._attrs['updatedAt']

    @property
    def url(self):
        return self.client.app_url + '/'.join([self.entity, self.project, 'reports', '--'.join([urllib.parse.quote(self.display_name.replace(' ', '-')), self.id.replace('=', '')])])

    def to_html(self, height=1024, hidden=False):
        """Generate HTML containing an iframe displaying this report."""
        url = self.url + '?jupyter=true'
        style = f'border:none;width:100%;height:{height}px;'
        prefix = ''
        if hidden:
            style += 'display:none;'
            prefix = ipython.toggle_button('report')
        return prefix + f'<iframe src={url!r} style={style!r}></iframe>'

    def _repr_html_(self) -> str:
        return self.to_html()