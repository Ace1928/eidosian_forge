import prometheus_client
from ..views import BaseHandler
class Healthcheck(BaseHandler):

    async def get(self):
        self.write('OK')