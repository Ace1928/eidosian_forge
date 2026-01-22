from neutronclient._i18n import _
def args2body_dns_update(parsed_args, resource, attribute):
    destination = 'dns_%s' % attribute
    no_destination = 'no_dns_%s' % attribute
    argument = getattr(parsed_args, destination)
    no_argument = getattr(parsed_args, no_destination)
    if argument:
        resource[destination] = argument
    if no_argument:
        resource[destination] = ''