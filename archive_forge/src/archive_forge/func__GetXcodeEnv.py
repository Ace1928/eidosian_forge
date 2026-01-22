import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _GetXcodeEnv(xcode_settings, built_products_dir, srcroot, configuration, additional_settings=None):
    """Return the environment variables that Xcode would set. See
  http://developer.apple.com/library/mac/#documentation/DeveloperTools/Reference/XcodeBuildSettingRef/1-Build_Setting_Reference/build_setting_ref.html#//apple_ref/doc/uid/TP40003931-CH3-SW153
  for a full list.

  Args:
      xcode_settings: An XcodeSettings object. If this is None, this function
          returns an empty dict.
      built_products_dir: Absolute path to the built products dir.
      srcroot: Absolute path to the source root.
      configuration: The build configuration name.
      additional_settings: An optional dict with more values to add to the
          result.
  """
    if not xcode_settings:
        return {}
    spec = xcode_settings.spec
    env = {'BUILT_FRAMEWORKS_DIR': built_products_dir, 'BUILT_PRODUCTS_DIR': built_products_dir, 'CONFIGURATION': configuration, 'PRODUCT_NAME': xcode_settings.GetProductName(), 'SRCROOT': srcroot, 'SOURCE_ROOT': '${SRCROOT}', 'TARGET_BUILD_DIR': built_products_dir, 'TEMP_DIR': '${TMPDIR}', 'XCODE_VERSION_ACTUAL': XcodeVersion()[0]}
    if xcode_settings.GetPerConfigSetting('SDKROOT', configuration):
        env['SDKROOT'] = xcode_settings._SdkPath(configuration)
    else:
        env['SDKROOT'] = ''
    if xcode_settings.mac_toolchain_dir:
        env['DEVELOPER_DIR'] = xcode_settings.mac_toolchain_dir
    if spec['type'] in ('executable', 'static_library', 'shared_library', 'loadable_module'):
        env['EXECUTABLE_NAME'] = xcode_settings.GetExecutableName()
        env['EXECUTABLE_PATH'] = xcode_settings.GetExecutablePath()
        env['FULL_PRODUCT_NAME'] = xcode_settings.GetFullProductName()
        mach_o_type = xcode_settings.GetMachOType()
        if mach_o_type:
            env['MACH_O_TYPE'] = mach_o_type
        env['PRODUCT_TYPE'] = xcode_settings.GetProductType()
    if xcode_settings._IsBundle():
        env['BUILT_FRAMEWORKS_DIR'] = os.path.join(built_products_dir + os.sep + xcode_settings.GetBundleFrameworksFolderPath())
        env['CONTENTS_FOLDER_PATH'] = xcode_settings.GetBundleContentsFolderPath()
        env['EXECUTABLE_FOLDER_PATH'] = xcode_settings.GetBundleExecutableFolderPath()
        env['UNLOCALIZED_RESOURCES_FOLDER_PATH'] = xcode_settings.GetBundleResourceFolder()
        env['JAVA_FOLDER_PATH'] = xcode_settings.GetBundleJavaFolderPath()
        env['FRAMEWORKS_FOLDER_PATH'] = xcode_settings.GetBundleFrameworksFolderPath()
        env['SHARED_FRAMEWORKS_FOLDER_PATH'] = xcode_settings.GetBundleSharedFrameworksFolderPath()
        env['SHARED_SUPPORT_FOLDER_PATH'] = xcode_settings.GetBundleSharedSupportFolderPath()
        env['PLUGINS_FOLDER_PATH'] = xcode_settings.GetBundlePlugInsFolderPath()
        env['XPCSERVICES_FOLDER_PATH'] = xcode_settings.GetBundleXPCServicesFolderPath()
        env['INFOPLIST_PATH'] = xcode_settings.GetBundlePlistPath()
        env['WRAPPER_NAME'] = xcode_settings.GetWrapperName()
    install_name = xcode_settings.GetInstallName()
    if install_name:
        env['LD_DYLIB_INSTALL_NAME'] = install_name
    install_name_base = xcode_settings.GetInstallNameBase()
    if install_name_base:
        env['DYLIB_INSTALL_NAME_BASE'] = install_name_base
    xcode_version, _ = XcodeVersion()
    if xcode_version >= '0500' and (not env.get('SDKROOT')):
        sdk_root = xcode_settings._SdkRoot(configuration)
        if not sdk_root:
            sdk_root = xcode_settings._XcodeSdkPath('')
        if sdk_root is None:
            sdk_root = ''
        env['SDKROOT'] = sdk_root
    if not additional_settings:
        additional_settings = {}
    else:
        for k in additional_settings:
            if not isinstance(additional_settings[k], str):
                additional_settings[k] = ' '.join(additional_settings[k])
    additional_settings.update(env)
    for k in additional_settings:
        additional_settings[k] = _NormalizeEnvVarReferences(additional_settings[k])
    return additional_settings