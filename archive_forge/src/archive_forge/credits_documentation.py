from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.fx.resize import resize
from moviepy.video.VideoClip import ImageClip, TextClip


    Parameters
    -----------
    
    creditfile
      A text file whose content must be as follows: ::
        
        # This is a comment
        # The next line says : leave 4 blank lines
        .blank 4
        
        ..Executive Story Editor
        MARCEL DURAND
        
        ..Associate Producers
        MARTIN MARCEL
        DIDIER MARTIN
        
        ..Music Supervisor
        JEAN DIDIER
    
    width
      Total width of the credits text in pixels
      
    gap
      Horizontal gap in pixels between the jobs and the names
    
    color
      Color of the text. See ``TextClip.list('color')``
      for a list of acceptable names.

    font
      Name of the font to use. See ``TextClip.list('font')`` for
      the list of fonts you can use on your computer.

    fontsize
      Size of font to use

    stroke_color
      Color of the stroke (=contour line) of the text. If ``None``,
      there will be no stroke.

    stroke_width
      Width of the stroke, in pixels. Can be a float, like 1.5.
    
        
    Returns
    ---------
    
    image
      An ImageClip instance that looks like this and can be scrolled
      to make some credits:

          Executive Story Editor    MARCEL DURAND
             Associate Producers    MARTIN MARCEL
                                    DIDIER MARTIN
                Music Supervisor    JEAN DIDIER
              
    