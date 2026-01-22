from libcloud.common.base import XmlResponse, ConnectionKey
class ZonomiResponse(XmlResponse):
    errors = None
    objects = None

    def __init__(self, response, connection):
        self.errors = []
        super().__init__(response=response, connection=connection)
        self.objects, self.errors = self.parse_body_and_errors()
        if self.errors:
            raise self._make_excp(self.errors[0])

    def parse_body_and_errors(self):
        error_dict = {}
        actions = None
        result_counts = None
        action_childrens = None
        data = []
        errors = []
        xml_body = super().parse_body()
        if xml_body.text is not None and xml_body.tag == 'error':
            error_dict['ERRORCODE'] = self.status
            if xml_body.text.startswith('ERROR: No zone found for'):
                error_dict['ERRORCODE'] = '404'
                error_dict['ERRORMESSAGE'] = 'Not found.'
            else:
                error_dict['ERRORMESSAGE'] = xml_body.text
            errors.append(error_dict)
        children = list(xml_body)
        if len(children) == 3:
            result_counts = children[1]
            actions = children[2]
        if actions is not None:
            actions_childrens = list(actions)
            action = actions_childrens[0]
            action_childrens = list(action)
        if action_childrens is not None:
            for child in action_childrens:
                if child.tag == 'zone' or child.tag == 'record':
                    data.append(child.attrib)
        if result_counts is not None and result_counts.attrib.get('deleted') == '1':
            data.append('DELETED')
        if result_counts is not None and result_counts.attrib.get('deleted') == '0' and (action.get('action') == 'DELETE'):
            error_dict['ERRORCODE'] = self.status
            error_dict['ERRORMESSAGE'] = 'Record not deleted.'
            errors.append(error_dict)
        return (data, errors)

    def success(self):
        return len(self.errors) == 0

    def _make_excp(self, error):
        """
        :param error: contains error code and error message
        :type error: dict
        """
        return ZonomiException(error['ERRORCODE'], error['ERRORMESSAGE'])