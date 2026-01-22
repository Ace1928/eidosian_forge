from designateclient.v2.base import V2Controller
class FloatingIPController(V2Controller):

    def set(self, floatingip_id, ptrdname, description=None, ttl=None):
        data = {'ptrdname': ptrdname}
        if description is not None:
            data['description'] = description
        if ttl is not None:
            data['ttl'] = ttl
        url = f'/reverse/floatingips/{floatingip_id}'
        return self._patch(url, data=data)

    def list(self, criterion=None):
        url = self.build_url('/reverse/floatingips', criterion)
        return self._get(url, response_key='floatingips')

    def get(self, floatingip_id):
        url = f'/reverse/floatingips/{floatingip_id}'
        return self._get(url)

    def unset(self, floatingip_id):
        data = {'ptrdname': None}
        url = f'/reverse/floatingips/{floatingip_id}'
        return self._patch(url, data=data)