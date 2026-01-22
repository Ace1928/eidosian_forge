from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.transfer.appliances import flags
def apply_args_to_order(order_resource, args, appliance_name=None):
    """Maps command arguments to appliance resource values.

  Args:
    order_resource (messages.Order): The target order resource.
    args (parser_extensions.Namespace): The args from the command.
    appliance_name (str): The name of the appliance associated with the order.

  Returns:
    List['field1', 'field2']
  """
    update_mask = []
    if args.IsSpecified('delivery_notes'):
        order_resource.deliveryNotes = args.delivery_notes
        update_mask.append('deliveryNotes')
    if args.IsSpecified('display_name'):
        order_resource.displayName = args.display_name
        update_mask.append('displayName')
    if appliance_name is not None:
        order_resource.appliances = [appliance_name]
    if args.address is not None:
        order_resource.address = {'addressLines': args.address.get('lines', None), 'locality': args.address.get('locality', None), 'administrativeArea': args.address.get('administrative-area', None), 'postalCode': args.address.get('postal-code', None), 'regionCode': _get_region_code(order_resource, args)}
        update_mask.append('address')
    if args.order_contact is not None:
        order_resource.orderContact = _apply_args_to_order_contact(args.order_contact)
        update_mask.append('orderContact')
    if args.shipping_contact is not None:
        order_resource.shippingContact = _apply_args_to_order_contact(args.shipping_contact)
        update_mask.append('shippingContact')
    return ','.join(update_mask)