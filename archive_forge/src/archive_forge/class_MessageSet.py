from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
from googlecloudsdk.core import log
class MessageSet(ProtocolBuffer.ProtocolMessage):
    """A protocol message which contains other protocol messages.

  This class is a specially-crafted ProtocolMessage which represents a
  container storing other protocol messages.  The contained messages can be
  of any protocol message type which has been assigned a unique type
  identifier.  No two messages in the MessageSet may have the same type,
  but otherwise there are no restrictions on what you can put in it.  Accessing
  the stored messages involves passing the class objects representing the
  types you are looking for:
    assert myMessageSet.has(MyMessageType)
    message = myMessageSet.get(MyMessageType)
    message = myMessageSet.mutable(MyMessageType)
    myMessageSet.remove(MyMessageType)

  Message types designed to be stored in MessageSets must have unique
  32-bit type identifiers.  Give your message type an identifier like so:
    parsed message MyMessageType {
      enum TypeId {MESSAGE_TYPE_ID = 12345678};
  To insure that your type ID is unique, use one of your perforce change
  numbers.  Just make sure you only use your own numbers and that you don't
  use the same one twice.

  The wire format of a MessageSet looks like this:
     parsed message MessageSet {
       repeated group Item = 1 {
         required fixed32 type_id = 2;
         required message<RawMessage> message = 3;
       };
     };
  The MessageSet class provides a special interface around this format for
  the sake of ease-of-use and type safety.

  See message_set_unittest.proto and message_set_py_unittest.py for examples.
  """

    def __init__(self, contents=None):
        """Construct a new MessageSet, with optional starting contents
    in binary protocol buffer format."""
        self.items = dict()
        if contents is not None:
            self.MergeFromString(contents)

    def get(self, message_class):
        """Gets a message of the given type from the set.

    If the MessageSet contains no message of that type, a default instance
    of the class is returned.  This is done to match the behavior of the
    accessors normally generated for an optional field of a protocol message.
    This makes it easier to transition from a long list of optional fields
    to a MessageSet.

    The returned message should not be modified.
    """
        if message_class.MESSAGE_TYPE_ID not in self.items:
            return message_class()
        item = self.items[message_class.MESSAGE_TYPE_ID]
        if item.Parse(message_class):
            return item.message
        else:
            return message_class()

    def mutable(self, message_class):
        """Gets a mutable reference to a message of the given type in the set.

    If the MessageSet contains no message of that type, one is created and
    added to the set.
    """
        if message_class.MESSAGE_TYPE_ID not in self.items:
            message = message_class()
            self.items[message_class.MESSAGE_TYPE_ID] = Item(message, message_class)
            return message
        item = self.items[message_class.MESSAGE_TYPE_ID]
        if not item.Parse(message_class):
            item.SetToDefaultInstance(message_class)
        return item.message

    def has(self, message_class):
        """Checks if the set contains a message of the given type."""
        if message_class.MESSAGE_TYPE_ID not in self.items:
            return 0
        item = self.items[message_class.MESSAGE_TYPE_ID]
        return item.Parse(message_class)

    def has_unparsed(self, message_class):
        """Checks if the set contains an unparsed message of the given type.

    This differs from has() when the set contains a message of the given type
    with a parse error.  has() will return false when this is the case, but
    has_unparsed() will return true.  This is only useful for error checking.
    """
        return message_class.MESSAGE_TYPE_ID in self.items

    def GetTypeIds(self):
        """Return a list of all type ids in the set.

    Returns:
      [ cls1.MESSAGE_TYPE_ID, ... ] for each cls in the set.  The returned
      list does not contain duplicates.
    """
        return self.items.keys()

    def NumMessages(self):
        """Return the number of messages in the set.  For any set the following
    invariant holds:
      set.NumMessages() == len(set.GetTypeIds())

    Returns:
      number of messages in the set
    """
        return len(self.items)

    def remove(self, message_class):
        """Removes any message of the given type from the set."""
        if message_class.MESSAGE_TYPE_ID in self.items:
            del self.items[message_class.MESSAGE_TYPE_ID]

    def __getitem__(self, message_class):
        if message_class.MESSAGE_TYPE_ID not in self.items:
            raise KeyError(message_class)
        item = self.items[message_class.MESSAGE_TYPE_ID]
        if item.Parse(message_class):
            return item.message
        else:
            raise KeyError(message_class)

    def __setitem__(self, message_class, message):
        self.items[message_class.MESSAGE_TYPE_ID] = Item(message, message_class)

    def __contains__(self, message_class):
        return self.has(message_class)

    def __delitem__(self, message_class):
        self.remove(message_class)

    def __len__(self):
        return len(self.items)

    def MergeFrom(self, other):
        """Merges the messages from MessageSet 'other' into this set.

    If both sets contain messages of matching types, those messages will be
    individually merged by type.
    """
        assert other is not self
        for type_id, item in other.items.items():
            if type_id in self.items:
                self.items[type_id].MergeFrom(item)
            else:
                self.items[type_id] = item.Copy()

    def Equals(self, other):
        """Checks if two MessageSets are equal."""
        if other is self:
            return 1
        if len(self.items) != len(other.items):
            return 0
        for type_id, item in other.items.items():
            if type_id not in self.items:
                return 0
            if not self.items[type_id].Equals(item):
                return 0
        return 1

    def __eq__(self, other):
        return other is not None and other.__class__ == self.__class__ and self.Equals(other)

    def __ne__(self, other):
        return not self == other

    def IsInitialized(self, debug_strs=None):
        """Checks if all messages in this set have had all of their required fields
    set."""
        initialized = 1
        for item in self.items.values():
            if not item.IsInitialized(debug_strs):
                initialized = 0
        return initialized

    def ByteSize(self):
        """Gets the byte size of a protocol buffer representing this MessageSet."""
        n = 2 * len(self.items)
        for type_id, item in self.items.items():
            n += item.ByteSize(self, type_id)
        return n

    def ByteSizePartial(self):
        """Gets the byte size of a protocol buffer representing this MessageSet.
    Does not count missing required fields."""
        n = 2 * len(self.items)
        for type_id, item in self.items.items():
            n += item.ByteSizePartial(self, type_id)
        return n

    def Clear(self):
        """Removes all messages from the set."""
        self.items = dict()

    def OutputUnchecked(self, out):
        """Writes the MessageSet to the encoder 'out'."""
        for type_id, item in self.items.items():
            out.putVarInt32(TAG_BEGIN_ITEM_GROUP)
            item.OutputUnchecked(out, type_id)
            out.putVarInt32(TAG_END_ITEM_GROUP)

    def OutputPartial(self, out):
        """Writes the MessageSet to the encoder 'out'.
    Does not assume required fields are set."""
        for type_id, item in self.items.items():
            out.putVarInt32(TAG_BEGIN_ITEM_GROUP)
            item.OutputPartial(out, type_id)
            out.putVarInt32(TAG_END_ITEM_GROUP)

    def TryMerge(self, decoder):
        """Attempts to decode a MessageSet from the decoder 'd' and merge it
    with this one."""
        while decoder.avail() > 0:
            tag = decoder.getVarInt32()
            if tag == TAG_BEGIN_ITEM_GROUP:
                type_id, message = Item.Decode(decoder)
                if type_id in self.items:
                    self.items[type_id].MergeFrom(Item(message))
                else:
                    self.items[type_id] = Item(message)
                continue
            if tag == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError
            decoder.skipData(tag)

    def _CToASCII(self, output_format):
        if _net_proto___parse__python is None:
            return ProtocolBuffer.ProtocolMessage._CToASCII(self, output_format)
        else:
            return _net_proto___parse__python.ToASCII(self, 'MessageSetInternal', output_format)

    def ParseASCII(self, s):
        if _net_proto___parse__python is None:
            ProtocolBuffer.ProtocolMessage.ParseASCII(self, s)
        else:
            _net_proto___parse__python.ParseASCII(self, 'MessageSetInternal', s)

    def ParseASCIIIgnoreUnknown(self, s):
        if _net_proto___parse__python is None:
            ProtocolBuffer.ProtocolMessage.ParseASCIIIgnoreUnknown(self, s)
        else:
            _net_proto___parse__python.ParseASCIIIgnoreUnknown(self, 'MessageSetInternal', s)

    def __str__(self, prefix='', printElemNumber=0):
        text = ''
        for type_id, item in self.items.items():
            if item.message_class is None:
                text += '%s[%d] <\n' % (prefix, type_id)
                text += '%s  (%d bytes)\n' % (prefix, len(item.message))
                text += '%s>\n' % prefix
            else:
                text += '%s[%s] <\n' % (prefix, item.message_class.__name__)
                text += item.message.__str__(prefix + '  ', printElemNumber)
                text += '%s>\n' % prefix
        return text
    _PROTO_DESCRIPTOR_NAME = 'MessageSet'