import time
from boto.compat import json
class ServicePoliciesStatus(OptionStatus):

    def new_statement(self, arn, ip):
        """
        Returns a new policy statement that will allow
        access to the service described by ``arn`` by the
        ip specified in ``ip``.

        :type arn: string
        :param arn: The Amazon Resource Notation identifier for the
            service you wish to provide access to.  This would be
            either the search service or the document service.

        :type ip: string
        :param ip: An IP address or CIDR block you wish to grant access
            to.
        """
        return {'Effect': 'Allow', 'Action': '*', 'Resource': arn, 'Condition': {'IpAddress': {'aws:SourceIp': [ip]}}}

    def _allow_ip(self, arn, ip):
        if 'Statement' not in self:
            s = self.new_statement(arn, ip)
            self['Statement'] = [s]
            self.save()
        else:
            add_statement = True
            for statement in self['Statement']:
                if statement['Resource'] == arn:
                    for condition_name in statement['Condition']:
                        if condition_name == 'IpAddress':
                            add_statement = False
                            condition = statement['Condition'][condition_name]
                            if ip not in condition['aws:SourceIp']:
                                condition['aws:SourceIp'].append(ip)
            if add_statement:
                s = self.new_statement(arn, ip)
                self['Statement'].append(s)
            self.save()

    def allow_search_ip(self, ip):
        """
        Add the provided ip address or CIDR block to the list of
        allowable address for the search service.

        :type ip: string
        :param ip: An IP address or CIDR block you wish to grant access
            to.
        """
        arn = self.domain.search_service_arn
        self._allow_ip(arn, ip)

    def allow_doc_ip(self, ip):
        """
        Add the provided ip address or CIDR block to the list of
        allowable address for the document service.

        :type ip: string
        :param ip: An IP address or CIDR block you wish to grant access
            to.
        """
        arn = self.domain.doc_service_arn
        self._allow_ip(arn, ip)

    def _disallow_ip(self, arn, ip):
        if 'Statement' not in self:
            return
        need_update = False
        for statement in self['Statement']:
            if statement['Resource'] == arn:
                for condition_name in statement['Condition']:
                    if condition_name == 'IpAddress':
                        condition = statement['Condition'][condition_name]
                        if ip in condition['aws:SourceIp']:
                            condition['aws:SourceIp'].remove(ip)
                            need_update = True
        if need_update:
            self.save()

    def disallow_search_ip(self, ip):
        """
        Remove the provided ip address or CIDR block from the list of
        allowable address for the search service.

        :type ip: string
        :param ip: An IP address or CIDR block you wish to grant access
            to.
        """
        arn = self.domain.search_service_arn
        self._disallow_ip(arn, ip)

    def disallow_doc_ip(self, ip):
        """
        Remove the provided ip address or CIDR block from the list of
        allowable address for the document service.

        :type ip: string
        :param ip: An IP address or CIDR block you wish to grant access
            to.
        """
        arn = self.domain.doc_service_arn
        self._disallow_ip(arn, ip)