from ctypes import c_void_p, c_ubyte
from pyglet.image import ImageData, Animation, AnimationFrame
from pyglet.image.codecs import *
from pyglet.libs.darwin.cocoapy import cf, quartz, NSMakeRect
from pyglet.libs.darwin.cocoapy import cfnumber_to_number
from pyglet.libs.darwin.cocoapy import kCGImageAlphaPremultipliedLast
from pyglet.libs.darwin.cocoapy import kCGImagePropertyGIFDictionary
from pyglet.libs.darwin.cocoapy import kCGImagePropertyGIFDelayTime
class QuartzImageDecoder(ImageDecoder):

    def get_file_extensions(self):
        return ['.bmp', '.cur', '.gif', '.ico', '.jp2', '.jpg', '.jpeg', '.pcx', '.png', '.tga', '.tif', '.tiff', '.xbm', '.xpm']

    def get_animation_file_extensions(self):
        return ['.gif']

    def _get_pyglet_ImageData_from_source_at_index(self, sourceRef, index):
        imageRef = c_void_p(quartz.CGImageSourceCreateImageAtIndex(sourceRef, index, None))
        format = 'RGBA'
        rgbColorSpace = c_void_p(quartz.CGColorSpaceCreateDeviceRGB())
        bitsPerComponent = 8
        width = quartz.CGImageGetWidth(imageRef)
        height = quartz.CGImageGetHeight(imageRef)
        bytesPerRow = 4 * width
        bufferSize = height * bytesPerRow
        buffer = (c_ubyte * bufferSize)()
        bitmap = c_void_p(quartz.CGBitmapContextCreate(buffer, width, height, bitsPerComponent, bytesPerRow, rgbColorSpace, kCGImageAlphaPremultipliedLast))
        quartz.CGContextDrawImage(bitmap, NSMakeRect(0, 0, width, height), imageRef)
        quartz.CGImageRelease(imageRef)
        quartz.CGContextRelease(bitmap)
        quartz.CGColorSpaceRelease(rgbColorSpace)
        pitch = bytesPerRow
        return ImageData(width, height, format, buffer, -pitch)

    def decode(self, filename, file):
        if not file:
            file = open(filename, 'rb')
        file_bytes = file.read()
        data = c_void_p(cf.CFDataCreate(None, file_bytes, len(file_bytes)))
        sourceRef = c_void_p(quartz.CGImageSourceCreateWithData(data, None))
        image = self._get_pyglet_ImageData_from_source_at_index(sourceRef, 0)
        cf.CFRelease(data)
        cf.CFRelease(sourceRef)
        return image

    def decode_animation(self, filename, file):
        if not file:
            file = open(filename, 'rb')
        file_bytes = file.read()
        data = c_void_p(cf.CFDataCreate(None, file_bytes, len(file_bytes)))
        sourceRef = c_void_p(quartz.CGImageSourceCreateWithData(data, None))
        count = quartz.CGImageSourceGetCount(sourceRef)
        frames = []
        for index in range(count):
            duration = 0.1
            props = c_void_p(quartz.CGImageSourceCopyPropertiesAtIndex(sourceRef, index, None))
            if cf.CFDictionaryContainsKey(props, kCGImagePropertyGIFDictionary):
                gif_props = c_void_p(cf.CFDictionaryGetValue(props, kCGImagePropertyGIFDictionary))
                if cf.CFDictionaryContainsKey(gif_props, kCGImagePropertyGIFDelayTime):
                    duration = cfnumber_to_number(c_void_p(cf.CFDictionaryGetValue(gif_props, kCGImagePropertyGIFDelayTime)))
            cf.CFRelease(props)
            image = self._get_pyglet_ImageData_from_source_at_index(sourceRef, index)
            frames.append(AnimationFrame(image, duration))
        cf.CFRelease(data)
        cf.CFRelease(sourceRef)
        return Animation(frames)