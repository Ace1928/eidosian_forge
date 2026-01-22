from neutronclient._i18n import _
def args2body_dns_create(parsed_args, resource, attribute):
    destination = 'dns_%s' % attribute
    argument = getattr(parsed_args, destination)
    if argument:
        resource[destination] = argument