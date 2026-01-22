from glanceclient.v1.apiclient import base
def _list_by_image(self, image):
    image_id = base.getid(image)
    url = '/v1/images/%s/members' % image_id
    resp, body = self.client.get(url)
    out = []
    for member in body['members']:
        member['image_id'] = image_id
        out.append(ImageMember(self, member, loaded=True))
    return out