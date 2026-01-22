from yowsup.layers import YowParallelLayer
import time, logging, random
from yowsup.layers import YowLayer
from yowsup.layers.noise.layer import YowNoiseLayer
from yowsup.layers.noise.layer_noise_segments import YowNoiseSegmentsLayer
from yowsup.layers.auth                        import YowAuthenticationProtocolLayer
from yowsup.layers.coder                       import YowCoderLayer
from yowsup.layers.logger                      import YowLoggerLayer
from yowsup.layers.network                     import YowNetworkLayer
from yowsup.layers.protocol_messages           import YowMessagesProtocolLayer
from yowsup.layers.protocol_media              import YowMediaProtocolLayer
from yowsup.layers.protocol_acks               import YowAckProtocolLayer
from yowsup.layers.protocol_receipts           import YowReceiptProtocolLayer
from yowsup.layers.protocol_groups             import YowGroupsProtocolLayer
from yowsup.layers.protocol_presence           import YowPresenceProtocolLayer
from yowsup.layers.protocol_ib                 import YowIbProtocolLayer
from yowsup.layers.protocol_notifications      import YowNotificationsProtocolLayer
from yowsup.layers.protocol_iq                 import YowIqProtocolLayer
from yowsup.layers.protocol_contacts           import YowContactsIqProtocolLayer
from yowsup.layers.protocol_chatstate          import YowChatstateProtocolLayer
from yowsup.layers.protocol_privacy            import YowPrivacyProtocolLayer
from yowsup.layers.protocol_profiles           import YowProfilesProtocolLayer
from yowsup.layers.protocol_calls import YowCallsProtocolLayer
from yowsup.common.constants import YowConstants
from yowsup.layers.axolotl import AxolotlSendLayer, AxolotlControlLayer, AxolotlReceivelayer
from yowsup.profile.profile import YowProfile
import inspect
def _construct(self):
    logger.debug('Initializing stack')
    for s in self.__stack:
        if type(s) is tuple:
            logger.warn('Implicit declaration of parallel layers in a tuple is deprecated, pass a YowParallelLayer instead')
            inst = YowParallelLayer(s)
        elif inspect.isclass(s):
            if issubclass(s, YowLayer):
                inst = s()
            else:
                raise ValueError('Stack must contain only subclasses of YowLayer')
        elif issubclass(s.__class__, YowLayer):
            inst = s
        else:
            raise ValueError('Stack must contain only subclasses of YowLayer')
        logger.debug('Constructed %s' % inst)
        inst.setStack(self)
        self.__stackInstances.append(inst)
    for i in range(0, len(self.__stackInstances)):
        upperLayer = self.__stackInstances[i + 1] if i + 1 < len(self.__stackInstances) else None
        lowerLayer = self.__stackInstances[i - 1] if i > 0 else None
        self.__stackInstances[i].setLayers(upperLayer, lowerLayer)