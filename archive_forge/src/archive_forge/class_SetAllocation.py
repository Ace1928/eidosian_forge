from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_placement import version
class SetAllocation(command.Lister, version.CheckerMixin):
    """Replaces the set of resource allocation(s) for a given consumer.

    Note that this is a full replacement of the existing allocations. If you
    want to retain the existing allocations and add a new resource class
    allocation, you must specify all resource class allocations, old and new.

    From ``--os-placement-api-version 1.8`` it is required to specify
    ``--project-id`` and ``--user-id`` to set allocations. It is highly
    recommended to provide a ``--project-id`` and ``--user-id`` when setting
    allocations for accounting and data consistency reasons.

    Starting with ``--os-placement-api-version 1.12`` the API response
    contains the ``project_id`` and ``user_id`` of allocations which also
    appears in the CLI output.

    Starting with ``--os-placement-api-version 1.28`` a consumer generation is
    used which facilitates safe concurrent modification of an allocation.

    Starting with ``--os-placement-api-version 1.38`` it is required to specify
    ``--consumer-type`` to set allocations. It is helpful to provide a
    ``--consumer-type`` when setting allocations so that resource usages can be
    filtered on consumer types.
    """

    def get_parser(self, prog_name):
        parser = super(SetAllocation, self).get_parser(prog_name)
        parser.add_argument('uuid', metavar='<uuid>', help='UUID of the consumer')
        parser.add_argument('--allocation', metavar='<rp=resource-provider-id,resource-class-name=amount-of-resource-used>', action='append', default=[], help='Create (or update) an allocation of a resource class. Specify option multiple times to set multiple allocations.')
        parser.add_argument('--project-id', metavar='project_id', help='ID of the consuming project. This option is required starting from ``--os-placement-api-version 1.8``.', required=self.compare_version(version.ge('1.8')))
        parser.add_argument('--user-id', metavar='user_id', help='ID of the consuming user. This option is required starting from ``--os-placement-api-version 1.8``.', required=self.compare_version(version.ge('1.8')))
        parser.add_argument('--consumer-type', metavar='consumer_type', help='The type of the consumer. This option is required starting from ``--os-placement-api-version 1.38``.', required=self.compare_version(version.ge('1.38')))
        return parser

    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = BASE_URL + '/' + parsed_args.uuid
        supports_consumer_generation = self.compare_version(version.ge('1.28'))
        if supports_consumer_generation:
            payload = http.request('GET', url).json()
            consumer_generation = payload.get('consumer_generation')
        allocations = parse_allocations(parsed_args.allocation)
        if not allocations:
            raise exceptions.CommandError('At least one resource allocation must be specified')
        if self.compare_version(version.ge('1.12')):
            allocations = {rp: {'resources': resources} for rp, resources in allocations.items()}
        else:
            allocations = [{'resource_provider': {'uuid': rp}, 'resources': resources} for rp, resources in allocations.items()]
        payload = {'allocations': allocations}
        if supports_consumer_generation:
            payload['consumer_generation'] = consumer_generation
        if self.compare_version(version.ge('1.8')):
            payload['project_id'] = parsed_args.project_id
            payload['user_id'] = parsed_args.user_id
        elif parsed_args.project_id or parsed_args.user_id:
            self.log.warning('--project-id and --user-id options do not affect allocation for --os-placement-api-version less than 1.8')
        if self.compare_version(version.ge('1.38')):
            payload['consumer_type'] = parsed_args.consumer_type
        elif parsed_args.consumer_type:
            self.log.warning('--consumer-type option does not affect allocation for --os-placement-api-version less than 1.38')
        http.request('PUT', url, json=payload)
        resp = http.request('GET', url).json()
        per_provider = resp['allocations'].items()
        props = {}
        fields = ('resource_provider', 'generation', 'resources')
        if self.compare_version(version.ge('1.12')):
            fields += ('project_id', 'user_id')
            props['project_id'] = resp['project_id']
            props['user_id'] = resp['user_id']
        if self.compare_version(version.ge('1.38')):
            fields += ('consumer_type',)
            props['consumer_type'] = resp['consumer_type']
        allocs = [dict(resource_provider=k, **props, **v) for k, v in per_provider]
        rows = (utils.get_dict_properties(a, fields) for a in allocs)
        return (fields, rows)