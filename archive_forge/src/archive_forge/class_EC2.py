import fixtures
import uuid
class EC2(Base):

    def __init__(self, client, user_id, project_id):
        super(EC2, self).__init__(client)
        self.user_id = user_id
        self.project_id = project_id

    def setUp(self):
        super(EC2, self).setUp()
        self.ref = {'user_id': self.user_id, 'project_id': self.project_id}
        self.entity = self.client.ec2.create(**self.ref)
        self.addCleanup(self.client.ec2.delete, self.user_id, self.entity.access)