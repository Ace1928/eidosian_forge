from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six.moves.urllib import parse as urllib_parse
from mimetypes import MimeTypes
import os
import json
import traceback
class RabbitClient:

    def __init__(self, module):
        self.module = module
        self.params = module.params
        self.check_required_library()
        self.check_host_params()
        self.url = self.params['url']
        self.proto = self.params['proto']
        self.username = self.params['username']
        self.password = self.params['password']
        self.host = self.params['host']
        self.port = self.params['port']
        self.vhost = self.params['vhost']
        self.queue = self.params['queue']
        self.exchange = self.params['exchange']
        self.routing_key = self.params['routing_key']
        self.headers = self.params['headers']
        self.cafile = self.params['cafile']
        self.certfile = self.params['certfile']
        self.keyfile = self.params['keyfile']
        if self.host is not None:
            self.build_url()
        if self.cafile is not None:
            self.append_ssl_certs()
        self.connect_to_rabbitmq()

    def check_required_library(self):
        if not HAS_PIKA:
            self.module.fail_json(msg=missing_required_lib('pika'), exception=PIKA_IMP_ERR)

    def check_host_params(self):
        if self.params['url'] is not None and any((self.params[k] is not None for k in ['proto', 'host', 'port', 'password', 'username', 'vhost'])):
            self.module.fail_json(msg='url and proto, host, port, vhost, username or password cannot be specified at the same time.')
        if self.params['url'] is None and any((self.params[k] is None for k in ['proto', 'host', 'port', 'password', 'username', 'vhost'])):
            self.module.fail_json(msg='Connection parameters must be passed via url, or,  proto, host, port, vhost, username or password.')

    def append_ssl_certs(self):
        ssl_options = {}
        if self.cafile:
            ssl_options['cafile'] = self.cafile
        if self.certfile:
            ssl_options['certfile'] = self.certfile
        if self.keyfile:
            ssl_options['keyfile'] = self.keyfile
        self.url = self.url + '?ssl_options=' + urllib_parse.quote(json.dumps(ssl_options))

    @staticmethod
    def rabbitmq_argument_spec():
        return dict(url=dict(type='str'), proto=dict(type='str', choices=['amqp', 'amqps']), host=dict(type='str'), port=dict(type='int'), username=dict(type='str'), password=dict(type='str', no_log=True), vhost=dict(type='str'), queue=dict(type='str'))
    ' Consider some file size limits here '

    def _read_file(self, path):
        try:
            with open(path, 'rb') as file_handle:
                return file_handle.read()
        except IOError as e:
            self.module.fail_json(msg='Unable to open file %s: %s' % (path, to_native(e)))

    @staticmethod
    def _check_file_mime_type(path):
        mime = MimeTypes()
        return mime.guess_type(path)

    def build_url(self):
        self.url = '{0}://{1}:{2}@{3}:{4}/{5}'.format(self.proto, self.username, self.password, self.host, self.port, self.vhost)

    def connect_to_rabbitmq(self):
        """
        Function to connect to rabbitmq using username and password
        """
        try:
            parameters = pika.URLParameters(self.url)
        except Exception as e:
            self.module.fail_json(msg='URL malformed: %s' % to_native(e))
        try:
            self.connection = pika.BlockingConnection(parameters)
        except Exception as e:
            self.module.fail_json(msg='Connection issue: %s' % to_native(e))
        try:
            self.conn_channel = self.connection.channel()
        except pika.exceptions.AMQPChannelError as e:
            self.close_connection()
            self.module.fail_json(msg='Channel issue: %s' % to_native(e))

    def close_connection(self):
        try:
            self.connection.close()
        except pika.exceptions.AMQPConnectionError:
            pass

    def basic_publish(self):
        self.content_type = self.params.get('content_type')
        if self.params.get('body') is not None:
            args = dict(body=self.params.get('body'), properties=pika.BasicProperties(content_type=self.content_type, delivery_mode=1, headers=self.headers))
        if self.params.get('src') is not None and self.content_type == 'text/plain':
            self.content_type = RabbitClient._check_file_mime_type(self.params.get('src'))[0]
            self.headers.update(filename=os.path.basename(self.params.get('src')))
            args = dict(body=self._read_file(self.params.get('src')), properties=pika.BasicProperties(content_type=self.content_type, delivery_mode=1, headers=self.headers))
        elif self.params.get('src') is not None:
            args = dict(body=self._read_file(self.params.get('src')), properties=pika.BasicProperties(content_type=self.content_type, delivery_mode=1, headers=self.headers))
        try:
            if self.queue is None and self.exchange is None:
                result = self.conn_channel.queue_declare(queue='', durable=self.params.get('durable'), exclusive=self.params.get('exclusive'), auto_delete=self.params.get('auto_delete'))
                self.conn_channel.confirm_delivery()
                self.queue = result.method.queue
            elif self.queue is not None and self.exchange is None:
                self.conn_channel.queue_declare(queue=self.queue, durable=self.params.get('durable'), exclusive=self.params.get('exclusive'), auto_delete=self.params.get('auto_delete'))
                self.conn_channel.confirm_delivery()
        except Exception as e:
            self.module.fail_json(msg='Queue declare issue: %s' % to_native(e))
        if self.routing_key is not None:
            args['routing_key'] = self.routing_key
        elif self.routing_key is None and self.queue is not None:
            args['routing_key'] = self.queue
        elif self.routing_key is None and self.exchange is not None:
            args['routing_key'] = self.exchange
        else:
            args['routing_key'] = ''
        if self.exchange is None:
            args['exchange'] = ''
        else:
            args['exchange'] = self.exchange
            if self.routing_key is None:
                args['routing_key'] = self.exchange
        try:
            self.conn_channel.basic_publish(**args)
            return True
        except pika.exceptions.UnroutableError:
            return False