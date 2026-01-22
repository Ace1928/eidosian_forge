from openstack import exceptions
from openstack import resource
from openstack import utils
def add_router_to_agent(self, session, router):
    body = {'router_id': router}
    url = utils.urljoin(self.base_path, self.id, 'l3-routers')
    resp = session.post(url, json=body)
    return resp.json()