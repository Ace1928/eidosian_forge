from pyomo.common.config import ConfigBlock, ConfigList, ConfigValue
class Default_Config(object):

    def config_block(self, init=False):
        config, blocks = minlp_config_block(init=init)
        return (config, blocks)